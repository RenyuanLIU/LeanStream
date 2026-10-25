"""Anonymous, read-only acceptance check for a published source/asset release.

Downloads and logs go only to a new caller-selected directory. This never
creates releases/tags, changes manifests, or runs inference/accuracy evaluation.
"""
import argparse,hashlib,json,os,re,subprocess,sys,tarfile,urllib.error,urllib.parse,urllib.request
from pathlib import Path,PurePosixPath
if __package__:
    from .common import sha,write,MODEL_REPO,MODEL_REVISION
    from .download_assets import OPENER,http_download
    from .archives import PUBLIC_REPOSITORY,ASSET_TAG
else:
    from common import sha,write,MODEL_REPO,MODEL_REVISION
    from download_assets import OPENER,http_download
    from archives import PUBLIC_REPOSITORY,ASSET_TAG

REQUIRED=[
    'LICENSE','CMakeLists.txt','include/leanstream/runtime.h','include/leanstream/weight_manager.h',
    'src/runtime/runtime.cpp','src/io/stream_io.cpp','predictor/predictor_cpu.cpp',
    'backends/triton/core.py','backends/triton/persistent.py','backends/triton/reduction.py','backends/triton/enqueue.cpp',
    'examples/ministral/reproduce/baselines/recalibrated.py','docs/media/autoregressive/library_visit.gif',
    'scripts/build_ministral.py','examples/ministral/run.py','examples/ministral/numerical.py',
    'examples/ministral/reproduce/artifact_catalog.json','examples/ministral/reproduce/asset_manifest.json',
    'examples/ministral/reproduce/scoring/coqa.py','examples/ministral/reproduce/coqa_selection.json',
    'examples/ministral/reproduce/baselines/fixed_cache.py','examples/ministral/reproduce/baselines/fixed_cache.cpp',
]
CORE_PREFIXES=('include/leanstream/','src/','predictor/','backends/','kernels/',
               'examples/ministral/python/','examples/ministral/native/','examples/ministral/pack/',
               'examples/ministral/reproduce/baselines/')
CLI={
    'examples.ministral.reproduce.download_model':['--output','--verify-only'],
    'examples.ministral.reproduce.download_assets':['--output','--model','--verify-only','--scope'],
    'examples.ministral.reproduce.prepare_coqa':['--assets','--download'],
    'examples.ministral.reproduce.verify_assets':['--assets','--model','--scope'],
    'examples.ministral.reproduce.build_baselines':['--output'],
    'examples.ministral.reproduce.smoke':['--assets','--model','--build','--numerical','--method','--output'],
    'examples.ministral.reproduce.reproduce_speed':['--assets','--model','--build','--method','--output'],
    'examples.ministral.reproduce.reproduce_accuracy':['--assets','--model','--build','--output'],
}

def require(value,message):
    if not value:raise ValueError(message)

def request_json(url):
    # Deliberately never consult GH_TOKEN, gh credentials, or an authenticated opener.
    request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'LeanStream-publication-check/1'})
    try:
        with OPENER.open(request,timeout=60) as response:return json.load(response)
    except urllib.error.HTTPError as exc:raise ValueError(f'Public endpoint unavailable (HTTP {exc.code}): {url}') from exc

def public_metadata(repository,tag,api=request_json,checks=None):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository),'Invalid repository')
    root='https://api.github.com/repos/'+repository
    info=api(root);require(info.get('private') is False and info.get('visibility')=='public','Repository is not verified PUBLIC')
    if checks is not None:checks['public_repository']='PASS'
    ref=api(root+'/git/ref/tags/'+urllib.parse.quote(tag,safe=''));obj=ref['object']
    for _ in range(8):
        if obj['type']=='commit':break
        require(obj['type']=='tag','Tag does not resolve to a commit')
        obj=api(root+'/git/tags/'+obj['sha'])['object']
    require(obj['type']=='commit' and re.fullmatch('[0-9a-f]{40}',obj['sha']),'Unresolved source tag')
    if checks is not None:checks['public_source_tag']='PASS'
    release=api(root+'/releases/tags/'+urllib.parse.quote(tag,safe=''))
    require(release.get('draft') is False and release.get('published_at') and release.get('tag_name')==tag,'Release is not published; draft/private preview does not pass')
    if checks is not None:checks['published_release']='PASS'
    assets=release['assets'];require(len({a['name'] for a in assets})==len(assets),'Duplicate Release attachment names')
    require(all(a.get('state')=='uploaded' for a in assets),'An attachment is not fully uploaded')
    return obj['sha'],release,{a['name']:a for a in assets}

def fetch_file(url,target,limit):
    require(url.startswith('https://'),'HTTPS required')
    request=urllib.request.Request(url,headers={'User-Agent':'LeanStream-publication-check/1'})
    total=0;temporary=target.with_name(target.name+'.part')
    with OPENER.open(request,timeout=60) as response,temporary.open('wb') as dest:
        while True:
            block=response.read(1<<20)
            if not block:break
            total+=len(block);require(total<=limit,'Download exceeds safety limit');dest.write(block)
    temporary.replace(target)

def extract_source(archive,output):
    seen=set();prefix=None;total=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            p=PurePosixPath(member.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in member.name and p.parts,'Unsafe source archive path')
            if prefix is None:prefix=p.parts[0]
            require(p.parts[0]==prefix,'Multiple source archive roots')
            require(member.isdir() or member.isfile(),'Source archive links/special files rejected')
            if len(p.parts)==1:
                require(member.isdir(),'Invalid source archive root');continue
            relative=Path(*p.parts[1:]);target=output/relative
            require(not target.is_symlink() and output.resolve() in target.resolve().parents,'Source path escapes destination')
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            require(str(relative) not in seen and member.size<=10<<20,'Duplicate/oversized source member');seen.add(str(relative))
            total+=member.size;require(total<=256<<20,'Source archive expands beyond limit')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                require(target.is_file() and target.stat().st_size==member.size,'Existing source member differs')
                digest=hashlib.sha256()
                with tar.extractfile(member) as src:
                    for block in iter(lambda:src.read(1<<20),b''):digest.update(block)
                require(sha(target)==digest.hexdigest(),'Existing source checksum differs');continue
            with tar.extractfile(member) as src,target.open('xb') as dest:
                for block in iter(lambda:src.read(1<<20),b''):dest.write(block)
    require({str(f.relative_to(output)) for f in output.rglob('*') if f.is_file() or f.is_symlink()}==seen,'Unexpected files in source acceptance tree')
    require(all((output/name).is_file() for name in REQUIRED),'Tagged source lacks required runtime/reproduction files')
    forbidden={'RELEASE_CHECKLIST.md','FINAL_AUDIT.json','AGENT_REPORT.md','IMPLEMENTATION_LOG.md'}
    require(not any(p.name in forbidden for p in output.rglob('*')),'Tagged source contains internal release records')

def approved_manifest(source,repository,tag,assets):
    directory=source/'examples/ministral/reproduce';raw=(directory/'asset_manifest.json').read_bytes();m=json.loads(raw);catalog=json.loads((directory/'artifact_catalog.json').read_text())
    require(m['publication']['status']=='PUBLISHED' and m['redistribution']=='APPROVED','Manifest is not approved/published; do not change legal state to bypass this check')
    require(m['publication']['repository']==repository and m['publication']['tag']==tag,'Manifest publication destination differs')
    require(m['source_repository']=='https://github.com/'+repository,'Manifest source repository differs')
    require(m['catalog_sha256']==sha(directory/'artifact_catalog.json'),'Tagged catalog hash differs')
    require(m['checkpoint']==dict(repo_id=MODEL_REPO,revision=MODEL_REVISION) and catalog['model']['repo_id']==MODEL_REPO and catalog['model']['revision']==MODEL_REVISION,'Unsupported checkpoint identity')
    require(len(m['archives'])==catalog['release_archive_count']==9 and len({r['name'] for r in m['archives']})==9,'Expected nine distinct v2 release archives')
    expected={r['name']:dict(size=r['size'],sha256=r['sha256']) for r in m['archives']}
    expected['asset_manifest.json']=dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    for name,item in expected.items():
        require(Path(name).name==name and name in assets and assets[name]['size']==item['size'],'Missing/wrong-size Release attachment: '+name)
    require('SHA256SUMS' in assets,'Missing SHA256SUMS')
    return m,expected

def verify_source_basis(repository,commit,manifest,api=request_json):
    base=manifest['source_base_revision'];require(re.fullmatch('[0-9a-f]{40}',base),'Invalid supported source revision')
    root='https://api.github.com/repos/'+repository
    comparison=api(root+'/compare/'+base+'...'+commit)
    require(comparison['status'] in ['ahead','identical'],'Tagged source does not descend from supported source revision')
    trees=[]
    for revision in [base,commit]:
        tree=api(root+'/git/commits/'+revision)['tree']['sha'];data=api(root+'/git/trees/'+tree+'?recursive=1')
        require(not data.get('truncated'),'Cannot verify a truncated source tree')
        trees.append({r['path']:r['sha'] for r in data['tree'] if r['type']=='blob' and (r['path'].startswith(CORE_PREFIXES) or r['path'] in ['examples/ministral/run.py','examples/ministral/numerical.py'])})
    require(trees[0] and trees[0]==trees[1],'Validated inference source differs from the supported source basis')

def anonymous_environment(source,output):
    env=dict(os.environ)
    for name in ['GH_TOKEN','GITHUB_TOKEN','GH_ENTERPRISE_TOKEN','GITHUB_ENTERPRISE_TOKEN','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_TOKEN_PATH','TRANSFORMERS_CACHE']:env.pop(name,None)
    env.update(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(output/'hf-cache'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_XET='1',USE_TF='0')
    return env

def read_sums(path):
    entries={}
    for line in path.read_text().splitlines():
        match=re.fullmatch(r'([0-9a-f]{64}) [ *]([^/\\]+)',line);require(match is not None,'Malformed SHA256SUMS')
        digest,name=match.groups();require(name not in entries and name not in ['.','..'],'Duplicate/unsafe checksum entry');entries[name]=digest
    return entries

def verify_sums(path,cache,expected):
    entries=read_sums(path)
    require(set(entries)==set(expected),'SHA256SUMS must cover the complete downloaded collection')
    for name,item in expected.items():require(entries[name]==item['sha256'] and (cache/name).stat().st_size==item['size'] and sha(cache/name)==entries[name],'Checksum mismatch: '+name)

def run(repository,tag,output):
    output=output.resolve()
    if output.exists():
        marker=output/'publication_check.json'
        require(marker.is_file(),'Refusing to reuse an unknown existing output directory')
        previous=json.loads(marker.read_text());require(previous.get('repository')==repository and previous.get('tag')==tag and previous.get('anonymous') is True,'Existing acceptance output identity differs')
    else:output.mkdir(parents=True,exist_ok=False)
    report=dict(status='INCOMPLETE',repository=repository,tag=tag,anonymous=True,checks={})
    try:
        report['stage']='public repository, source tag and published Release'
        commit,release,assets=public_metadata(repository,tag,checks=report['checks'])
        if 'previous' in locals() and previous.get('source_commit'):require(previous['source_commit']==commit,'Source tag moved since previous acceptance attempt')
        report.update(source_commit=commit,release_id=release['id']);report['checks']['public_repository_tag_release']='PASS'
        report['stage']='tagged source and manifest approval'
        source=output/'source';fetch_file(f'https://api.github.com/repos/{repository}/tarball/{commit}',output/'source.tar.gz',64<<20);extract_source(output/'source.tar.gz',source)
        manifest,expected=approved_manifest(source,repository,tag,assets);verify_source_basis(repository,commit,manifest);report['checks']['tagged_source_and_supported_basis']='PASS'
        # The tagged source performs the real downloads into an initially empty cache.
        env=anonymous_environment(source,output)
        def command(label,args):
            with (output/(label+'.log')).open('w') as log:subprocess.run([sys.executable,'-B',*args],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for module,flags in CLI.items():
            label=module.rsplit('.',1)[-1]+'-help';command(label,['-m',module,'--help']);text=(output/(label+'.log')).read_text();require(all(flag in text for flag in flags),'Unsupported documented CLI: '+module)
        command('build-help',['scripts/build_ministral.py','--help']);require('--arch' in (output/'build-help.log').read_text(),'Build architecture flag missing');report['checks']['documented_cli_flags']='PASS'
        report['stage']='anonymous pinned checkpoint download'
        upstream=request_json(f'https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}');require(upstream['sha']==MODEL_REVISION,'Upstream pinned revision is not independently available')
        command('checkpoint',['-m','examples.ministral.reproduce.download_model','--output',str(output/'checkpoint')]);report['checks']['anonymous_checkpoint_download']='PASS'
        report['stage']='anonymous Release download and extraction'
        cache=output/'downloads'
        command('assets',['-m','examples.ministral.reproduce.download_assets','--repository',repository,'--release-tag',tag,'--output',str(output/'assets'),'--model',str(output/'checkpoint'),'--cache',str(cache)])
        url=assets['SHA256SUMS']['browser_download_url'];fetch_file(url,cache/'SHA256SUMS',1<<20);require((cache/'SHA256SUMS').stat().st_size==assets['SHA256SUMS']['size'],'SHA256SUMS attachment size mismatch')
        sums=read_sums(cache/'SHA256SUMS');require(set(expected)<=set(sums),'SHA256SUMS omits a required archive/manifest')
        for name in set(sums)-set(expected):
            require(name in assets and assets[name]['size']<=10<<20,'Unknown/oversized supplemental notice')
            expected[name]=dict(size=assets[name]['size'],sha256=sums[name])
            http_download(assets[name]['browser_download_url'],cache/name,expected[name])
        verify_sums(cache/'SHA256SUMS',cache,expected)
        command('verify',['-m','examples.ministral.reproduce.verify_assets','--assets',str(output/'assets'),'--model',str(output/'checkpoint'),'--scope','runtime'])
        report['checks'].update(anonymous_release_download='PASS',archive_checksums='PASS',fresh_catalog_extraction='PASS',checkpoint_identity='PASS')
        command('coqa',['-m','examples.ministral.reproduce.prepare_coqa','--assets',str(output/'assets'),'--download'])
        command('verify-all',['-m','examples.ministral.reproduce.verify_assets','--assets',str(output/'assets'),'--model',str(output/'checkpoint'),'--scope','all'])
        report['checks']['fixed500_reconstruction']='PASS'
        report.update(status='PASS',stage='complete',downloaded_asset_bytes=sum((cache/name).stat().st_size for name in [*expected,'SHA256SUMS']),inference_tests='NOT_RUN; run documented Jetson smoke commands separately')
    except (OSError,ValueError,KeyError,TypeError,IndexError,subprocess.CalledProcessError,tarfile.TarError) as exc:
        report.update(status='BLOCKED',error=str(exc))
    write(output/'publication_check.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',default=PUBLIC_REPOSITORY);p.add_argument('--release-tag',default=ASSET_TAG);p.add_argument('--output',type=Path,required=True,help='New empty output path; full acceptance downloads source, assets and upstream checkpoint');a=p.parse_args()
    try:report=run(a.repository,a.release_tag,a.output)
    except OSError as exc:p.exit(1,str(exc)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__':main()
