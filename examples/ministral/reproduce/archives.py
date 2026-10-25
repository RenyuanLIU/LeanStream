"""Versioned release archives tied to the original byte-level artifact catalog."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
if __package__:
    from .common import HERE,VERSION,sha,load,write
    from .assets import AssetError,safe_path
else:
    from common import HERE,VERSION,sha,load,write
    from assets import AssetError,safe_path

PUBLIC_REPOSITORY='RenyuanLIU/LeanStream'
ASSET_TAG='ministral3-8b-assets-v1'
LIMIT=1_700_000_000

def release_manifest(path,catalog=None):
    m=load(path);catalog=catalog or load(HERE/'artifact_catalog.json')
    if m.get('format_version')!=1 or m.get('version')!=VERSION or m.get('catalog_sha256')!=sha(HERE/'artifact_catalog.json'):
        raise AssetError('Release manifest/catalog identity mismatch')
    if m.get('checkpoint')!={k:v for k,v in catalog['model'].items() if k!='files'}:raise AssetError('Release checkpoint identity mismatch')
    if m.get('geometry')!=dict(layers=34,H=4096,I=14336,groups=7168,neurons_per_group=2):raise AssetError('Release geometry mismatch')
    if m.get('quantization')!=dict(bits=4,encoding='signed_twos_complement_low_nibble_first_-7_to_7',quant_group_size=128,scale_dtype='float16',projection_order=['gate','up','down']):raise AssetError('Release quantization metadata mismatch')
    if m.get('packed_manifest_sha256')!=catalog['files']['leanstream_weights/g2_weights/manifest.json']['sha256']:raise AssetError('Release packed-weight identity mismatch')
    seen=set();archive_names=set()
    for archive in m.get('archives',[]):
        name=archive['name'];safe_path(Path(path).parent,name)
        if name in archive_names or not archive['files']:raise AssetError('Duplicate/empty archive')
        archive_names.add(name)
        if Path(name).name!=name or not name.endswith('.tar.gz') or not 0<archive['size']<int(1.8*1024**3):raise AssetError('Invalid archive name/size')
        if len(archive['sha256'])!=64 or any(c not in '0123456789abcdef' for c in archive['sha256']):raise AssetError('Invalid archive checksum')
        for f in archive['files']:
            if f in seen or f not in catalog['files']:raise AssetError('Duplicate/unknown archive member')
            seen.add(f)
    deferred=set(m.get('reconstruct_files',[]))
    if seen & deferred or seen|deferred!=set(catalog['files']):raise AssetError('Incomplete archive inventory')
    if any(not n.startswith('benchmarks/') for n in deferred):raise AssetError('Only dataset-derived benchmark files may be deferred')
    return m

def extract_archive(path,record,output,catalog):
    path=Path(path);output=Path(output)
    if path.stat().st_size!=record['size'] or sha(path)!=record['sha256']:raise AssetError('Archive checksum/size mismatch: '+path.name)
    expected=set(record['files']);seen=set()
    with tarfile.open(path,mode='r|gz') as archive:
        for member in archive:
            name=member.name;target=safe_path(output,name)
            if name not in expected or name in seen or not member.isfile() or member.issym() or member.islnk():raise AssetError('Unsafe/duplicate/unexpected archive member: '+name)
            if target.is_symlink():raise AssetError('Symlink artifact target rejected')
            item=catalog['files'][name]
            if member.size!=item['size']:raise AssetError('Archive member size mismatch: '+name)
            seen.add(name)
            if target.exists():
                if not target.is_file() or target.stat().st_size!=item['size'] or sha(target)!=item['sha256']:raise AssetError('Incompatible existing artifact: '+name)
                continue
            target.parent.mkdir(parents=True,exist_ok=True);temporary=safe_path(output,name+'.part')
            if temporary.is_symlink():raise AssetError('Symlink partial artifact rejected')
            h=hashlib.sha256();total=0
            with archive.extractfile(member) as source,temporary.open('wb') as dest:
                for block in iter(lambda:source.read(8<<20),b''):dest.write(block);h.update(block);total+=len(block)
            if total!=item['size'] or h.hexdigest()!=item['sha256']:raise AssetError('Extracted file checksum mismatch: '+name)
            temporary.replace(target)
    if seen!=expected:raise AssetError('Archive is missing expected members')

def package(source,output,source_revision):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    catalog=load(HERE/'artifact_catalog.json');records=catalog['files'];deferred=sorted(n for n in records if n.startswith('benchmarks/'))
    categories={}
    for name,item in sorted(records.items()):
        path=safe_path(source,name)
        if not path.is_file() or path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:raise AssetError('Source is missing or incompatible: '+name)
        if name in deferred:continue
        category={'leanstream_weights':'g2-weights','full_predictors':'predictors-full','p1_lite_predictors':'predictors-p1-lite','baseline_weights':'baseline-weights'}.get(name.split('/')[0],'metadata')
        categories.setdefault(category,[]).append(name)
    archives=[]
    for category,names in sorted(categories.items()):
        parts=[];current=[];size=0
        for name in names:
            amount=records[name]['size']+2048
            if amount>LIMIT:raise AssetError('Individual file exceeds independent archive limit: '+name)
            if current and size+amount>LIMIT:parts.append(current);current=[];size=0
            current.append(name);size+=amount
        if current:parts.append(current)
        for index,members in enumerate(parts,1):
            name=f'ministral3-8b-{category}-part{index:02d}.tar.gz';path=output/name
            with path.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|',format=tarfile.GNU_FORMAT) as archive:
                for member in members:
                    item=records[member];info=tarfile.TarInfo(member);info.size=item['size'];info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                    with safe_path(source,member).open('rb') as f:archive.addfile(info,f)
            archives.append(dict(name=name,size=path.stat().st_size,sha256=sha(path),files=members))
            print(name,path.stat().st_size,flush=True)
    m=dict(format_version=1,version=VERSION,publication=dict(status='PRIVATE_REVIEW_ONLY',repository=PUBLIC_REPOSITORY,tag=ASSET_TAG,uploaded=False),redistribution='PENDING_OWNER_APPROVAL',catalog_sha256=sha(HERE/'artifact_catalog.json'),source_repository='https://github.com/'+PUBLIC_REPOSITORY,source_base_revision=source_revision,checkpoint={k:v for k,v in catalog['model'].items() if k!='files'},geometry=dict(layers=34,H=4096,I=14336,groups=7168,neurons_per_group=2),weight_format='leanstream_signed_int4_per128_g2_v0',quantization=dict(bits=4,encoding='signed_twos_complement_low_nibble_first_-7_to_7',quant_group_size=128,scale_dtype='float16',projection_order=['gate','up','down']),predictor_format='calibrated CPU W4A8 full d1664 / compact d512',packed_manifest_sha256=records['leanstream_weights/g2_weights/manifest.json']['sha256'],archives=archives,reconstruct_files=deferred,extracted_payload_bytes=sum(records[n]['size'] for n in records if n not in deferred),total_archive_bytes=sum(a['size'] for a in archives),compatibility=dict(platform='Jetson AGX Orin SM87',cuda='11.8',torch='vendor 2.1.0a0',triton='3.1.0',persistent_B_CTAs=32,warps_per_CTA=4,QD=24),logical_physical_mapping=dict(identity='original predictor Group2 permutation and per-layer logical_to_physical mapping',bound_by_packed_manifest_sha256=records['leanstream_weights/g2_weights/manifest.json']['sha256']),license_provenance=dict(model_declared_license='Apache-2.0',model_repository='https://huggingface.co/'+catalog['model']['repo_id']+'/tree/'+catalog['model']['revision'],dataset_terms='https://stanfordnlp.github.io/coqa/#license',dataset_contents_in_archives=False,owner_approval_required=True),notices=['LICENSE_OR_NOTICE','LICENSES/Apache-2.0.txt'],dataset_policy='No CoQA text, answers or token IDs in archives. Reconstruct exact benchmark inputs from the pinned official dataset using prepare_coqa.py.')
    write(output/'asset_manifest.json',m)
    (output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in archives)+sha(output/'asset_manifest.json')+'  asset_manifest.json\n')
    (output/'release_notes.md').write_text('# Ministral-3 8B assets v1\n\nNot published. Model-derived weights and calibrated predictors require owner redistribution approval. Apache-2.0 text is included; derivative provenance and applicable upstream notices must be reviewed. No upstream checkpoint or CoQA dataset contents are republished. Download the pinned checkpoint separately; reconstruct original benchmark inputs with the public deterministic recipe. Source base revision: '+source_revision+'\n\nTotal compressed bytes: '+str(m['total_archive_bytes'])+'\n')
    release_manifest(output/'asset_manifest.json',catalog);return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-revision',required=True);a=p.parse_args()
    package(a.source,a.output,a.source_revision)
