"""Download only the independently pinned upstream checkpoint; no conversion."""
import argparse
from pathlib import Path
if __package__:
    from .common import HERE,load,sha
    from .assets import AssetError
    from .download_assets import transfer
else:
    from common import HERE,load,sha
    from assets import AssetError
    from download_assets import transfer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();m=load(HERE/'artifact_catalog.json')['model']
    try:
        for name,item in m['files'].items():
            target=a.output/name
            if target.exists():
                if target.stat().st_size==item['size'] and sha(target)==item['sha256']:continue
                raise AssetError('Incompatible existing checkpoint file: '+name)
            if a.verify_only:raise AssetError('Missing checkpoint file: '+name)
            from huggingface_hub import hf_hub_download
            source=hf_hub_download(repo_id=m['repo_id'],revision=m['revision'],filename=name)
            transfer(source,target,item)
        print('PASS: pinned upstream checkpoint files verified; no model conversion')
    except (OSError,ValueError) as e:p.exit(1,str(e)+'\n')
if __name__=='__main__':main()
