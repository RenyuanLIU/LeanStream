"""Author-only byte-preserving copy of the sealed v2 runtime collection.

Normal users run download_assets. This does not train, convert, pack, or upload.
Dataset-derived benchmark files are deliberately excluded and reconstructed separately.
"""
import argparse,json
from pathlib import Path
if __package__:
    from .common import HERE,load,sha,write
    from .assets import descriptor,safe_path,AssetError
    from .download_assets import transfer
else:
    from common import HERE,load,sha,write
    from assets import descriptor,safe_path,AssetError
    from download_assets import transfer

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True,help='Complete verified v2 extracted runtime collection');p.add_argument('--output',type=Path,required=True);p.add_argument('--catalog-output',type=Path);a=p.parse_args()
    try:
        catalog=descriptor(HERE/'artifact_catalog.json')
        if load(a.source/'MANIFEST.json')!=catalog:raise AssetError('Source is not the sealed v2 collection; no substitutions accepted')
        a.output.mkdir(parents=True,exist_ok=False);copied=[]
        for name,item in catalog['files'].items():
            if name.startswith('benchmarks/'):continue
            transfer(safe_path(a.source,name),safe_path(a.output,name),item);copied.append(name)
        write(a.output/'MANIFEST.json',catalog)
        (a.output/'SHA256SUMS').write_text(''.join(catalog['files'][n]['sha256']+'  '+n+'\n' for n in copied)+sha(a.output/'MANIFEST.json')+'  MANIFEST.json\n')
        if a.catalog_output:write(a.catalog_output,catalog)
        print(json.dumps(dict(status='VERIFIED_RUNTIME_COPY',files=len(copied),version=catalog['version'],dataset_contents_copied=False)))
    except (OSError,ValueError,KeyError) as e:p.exit(1,str(e)+'\n')
if __name__=='__main__':main()
