"""Verify exact experiment files; no conversion, training, or downloads."""
import argparse
if __package__:
    from .assets import verify, AssetError
else:
    from assets import verify, AssetError

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--model');p.add_argument('--scope',choices=['all','runtime'],default='all');p.add_argument('--without-model',action='store_true',help='Author archive verification only; no inference claim');a=p.parse_args()
    try:
        m=verify(a.assets,model=not a.without_model,model_root=a.model,scope=a.scope)
        count=sum(1 for n in m['files'] if a.scope=='all' or not n.startswith('benchmarks/'))
        print(f"PASS: {count} requested artifact files; 34 layers; Group2/predictor/fixed-hot/YaRN identities; checkpoint files {'not requested' if a.without_model else 'verified'}; benchmark scope={a.scope}")
    except (AssetError,OSError,KeyError,ValueError) as e:p.exit(1,str(e)+'\n')
if __name__=='__main__':main()
