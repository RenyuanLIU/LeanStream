"""Pool measured decode wall time; never substitute historical performance."""
import argparse,csv
from pathlib import Path
if __package__:
    from .common import load,write
else:
    from common import load,write

def compare(root):
    root=Path(root);results=[]
    for method in ['block','one-shot','leanstream']:
        runs=[load(p) for p in sorted(root.glob(method+'-*/summary.json')) if not load(p).get('sanity')]
        if not runs:continue
        if len(runs)!=2 or {r['direction'] for r in runs}!={'forward','reverse'} or any(r['status']!='PASS' or r['method']!=method or r['decode_tokens']<=0 or r['total_decode_ms']<=0 for r in runs):
            raise ValueError('Incomplete or incompatible forward/reverse results for '+method)
        n=sum(x['decode_tokens'] for x in runs);total=sum(x['total_decode_ms'] for x in runs)
        results.append(dict(method=method,decode_tokens=n,total_decode_ms=total,ms_per_token=total/n,tokens_per_s=1000*n/total,directions=runs))
    if not results:raise ValueError('No measured runs to compare')
    write(root/'comparison.json',dict(status='COMPLETE',results=results,historical_only_tokens_per_s=dict(block=2.9,one_shot=3.2,leanstream=9.0)))
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method','decode_tokens','total_decode_ms','ms_per_token','tokens_per_s']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in results)
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);compare(p.parse_args().output)
