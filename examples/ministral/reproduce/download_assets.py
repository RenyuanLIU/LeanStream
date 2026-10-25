"""Pinned GitHub Release archives, verified local extraction, and offline resume."""
import argparse,json,os,re,shutil,tarfile,urllib.request,urllib.parse
from pathlib import Path
if __package__:
    from .common import HERE,sha,write,load
    from .assets import descriptor,safe_path,AssetError,verify
    from .archives import PUBLIC_REPOSITORY,ASSET_TAG,release_manifest,extract_archive
else:
    from common import HERE,sha,write,load
    from assets import descriptor,safe_path,AssetError,verify
    from archives import PUBLIC_REPOSITORY,ASSET_TAG,release_manifest,extract_archive

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        redirected=super().redirect_request(request,fp,code,msg,headers,newurl)
        if urllib.parse.urlsplit(request.full_url).hostname!=urllib.parse.urlsplit(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected

OPENER=urllib.request.build_opener(SafeRedirect)

def transfer(source,target,record):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==record['size'] and sha(target)==record['sha256']:return
        raise AssetError('Existing file checksum mismatch; inspect/remove it explicitly: '+str(target))
    temporary=target.with_suffix(target.suffix+'.part')
    if target.is_symlink() or temporary.is_symlink():raise AssetError('Symlink download destination rejected')
    shutil.copyfile(source,temporary)
    if temporary.stat().st_size!=record['size'] or sha(temporary)!=record['sha256']:raise AssetError('Downloaded checksum mismatch: '+str(target))
    temporary.replace(target)

def http_download(url,target,record,token=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not url.startswith('https://'):raise AssetError('HTTPS download required')
    if target.exists():
        if target.stat().st_size==record['size'] and sha(target)==record['sha256']:return
        raise AssetError('Incompatible cached download: '+target.name)
    part=target.with_name(target.name+'.part')
    if target.is_symlink() or part.is_symlink():raise AssetError('Symlink download destination rejected')
    offset=part.stat().st_size if part.exists() else 0
    if offset>record['size']:raise AssetError('Partial download exceeds expected size: '+target.name)
    if offset==record['size']:
        if sha(part)!=record['sha256']:raise AssetError('Partial download checksum mismatch: '+target.name)
        part.replace(target);return
    headers={'User-Agent':'LeanStream-reproduction/1','Accept':'application/octet-stream'}
    if token:headers['Authorization']='Bearer '+token
    if offset:headers['Range']=f'bytes={offset}-'
    request=urllib.request.Request(url,headers=headers)
    with OPENER.open(request,timeout=60) as response:
        status=response.getcode()
        if status==206:
            if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise AssetError('Incorrect resume range')
            mode='ab'
        elif status==200:mode='wb';offset=0
        else:raise AssetError('Unexpected download status: '+str(status))
        with part.open(mode) as f:
            total=offset
            while True:
                block=response.read(8<<20)
                if not block:break
                total+=len(block)
                if total>record['size']:raise AssetError('Download exceeds declared archive size')
                f.write(block)
    if part.stat().st_size!=record['size'] or sha(part)!=record['sha256']:raise AssetError('Download checksum/size mismatch: '+target.name)
    part.replace(target)

def release_urls(repository,tag,token=None,release_id=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):raise AssetError('Invalid release repository')
    headers={'User-Agent':'LeanStream-reproduction/1','Accept':'application/vnd.github+json'}
    if token:headers['Authorization']='Bearer '+token
    if release_id is not None and not token:raise AssetError('Authenticated private draft access requires GH_TOKEN or GITHUB_TOKEN')
    suffix=str(release_id) if release_id is not None else 'tags/'+urllib.parse.quote(tag,safe='')
    url=f'https://api.github.com/repos/{repository}/releases/{suffix}'
    with OPENER.open(urllib.request.Request(url,headers=headers),timeout=30) as response:m=json.load(response)
    # Authenticated private previews use the asset API; public users use ordinary HTTPS.
    return {r['name']:(r['url'] if token else r['browser_download_url']) for r in m['assets']}

def install_archives(output,archive_dir,manifest_path,catalog):
    m=release_manifest(manifest_path,catalog);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    existing=output/'MANIFEST.json'
    if existing.exists() and load(existing)!=catalog:raise AssetError('Output has an incompatible artifact manifest')
    for record in m['archives']:
        extract_archive(Path(archive_dir)/record['name'],record,output,catalog)
    write(existing,catalog)
    (output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {n}\n" for n,r in catalog['files'].items())+sha(existing)+'  MANIFEST.json\n')
    return m

def main():
    if __package__:
        from .assets import verify
    else:
        from assets import verify
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--descriptor',type=Path,default=HERE/'artifact_catalog.json')
    p.add_argument('--release-manifest',type=Path,default=HERE/'asset_manifest.json')
    p.add_argument('--repository',default=PUBLIC_REPOSITORY,help='Explicit private-preview override; final default is public LeanStream')
    p.add_argument('--release-tag',default=ASSET_TAG)
    p.add_argument('--release-id',type=int,help='Authenticated private Draft Release ID; not needed for public users')
    p.add_argument('--local-archives',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--model',type=Path,help='Separate pinned upstream checkpoint, never converted')
    p.add_argument('--scope',choices=['runtime','all'],default='runtime')
    p.add_argument('--local-source',type=Path,help='Existing exact unpacked collection, for author/offline use')
    p.add_argument('--model-source',type=Path,help='Legacy offline copy into output/model; prefer separate --model')
    a=p.parse_args()
    try:
        catalog=descriptor(a.descriptor);a.output=a.output.resolve()
        if a.verify_only:
            verify(a.output,model=bool(a.model or (a.output/'model').exists()),model_root=a.model,scope=a.scope);print('PASS: verified requested exact artifact identities');return
        if a.local_source:
            for name,item in catalog['files'].items():transfer(safe_path(a.local_source,name),safe_path(a.output,name),item)
            if a.model_source:
                for name,item in catalog['model']['files'].items():transfer(a.model_source/name,a.output/'model'/name,item)
            write(a.output/'MANIFEST.json',catalog)
            (a.output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {n}\n" for n,r in catalog['files'].items())+sha(a.output/'MANIFEST.json')+'  MANIFEST.json\n')
            # Preserve the existing offline collection workflow.
            if a.model_source:verify(a.output)
            else:verify(a.output,model=bool(a.model),model_root=a.model,scope=a.scope)
            print('PASS: installed exact local collection');return
        if not a.release_manifest.is_file():raise AssetError('Release packages are not prepared; no public assets are available')
        m=release_manifest(a.release_manifest,catalog)
        cache=a.cache or a.output.parent/'.leanstream-downloads'/m['version']
        if a.local_archives:archive_dir=a.local_archives
        else:
            if a.repository==PUBLIC_REPOSITORY and (m['publication']['status']!='PUBLISHED' or m.get('redistribution')!='APPROVED'):raise AssetError('Public asset Release is UNPUBLISHED. Use approved local archives for review; do not expect private assets to download anonymously.')
            token=os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
            urls=release_urls(a.repository,a.release_tag,token,a.release_id)
            manifest_record=dict(size=a.release_manifest.stat().st_size,sha256=sha(a.release_manifest))
            for name,item in [('asset_manifest.json',manifest_record)]+[(r['name'],r) for r in m['archives']]:
                if name not in urls:raise AssetError('Release is missing required asset: '+name)
                http_download(urls[name],cache/name,item,token)
            archive_dir=cache
        install_archives(a.output,archive_dir,a.release_manifest,catalog)
        verify(a.output,model=bool(a.model),model_root=a.model,scope='runtime')
        if a.scope=='all':verify(a.output,model=bool(a.model),model_root=a.model,scope='all')
        print('PASS: exact runtime artifacts extracted. Download the checkpoint separately and run prepare_coqa.py for historical benchmark inputs.')
    except (OSError,KeyError,ValueError,tarfile.TarError) as e:p.exit(1,str(e)+'\n')
if __name__=='__main__':main()
