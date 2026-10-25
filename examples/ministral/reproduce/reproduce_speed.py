"""Sequential matched E2E decode using the optional historical application policies."""
import argparse,subprocess,sys
from pathlib import Path
if __package__:
    from .common import HERE,environment,write
else:
    from common import HERE,environment,write
if __package__:
    from .assets import verify
else:
    from assets import verify
if __package__:
    from .compare_results import compare
else:
    from compare_results import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,help='Separate pinned upstream checkpoint directory');p.add_argument('--assets',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--method',choices=['block','one-shot','leanstream','all'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sanity',action='store_true',help='Original two-decode numerical/descriptor checks; baselines only');a=p.parse_args()
    verify(a.assets,model_root=a.model);a.output.mkdir(parents=True,exist_ok=False)
    methods=['block','one-shot','leanstream'] if a.method=='all' else [a.method]
    if a.sanity and 'leanstream' in methods:raise ValueError('Use numerical.py for LeanStream sanity')
    for direction in (['forward'] if a.sanity else ['forward','reverse']):
        for method in methods:
            cmd=[sys.executable,str(HERE/'worker.py'),'--assets',str(a.assets.resolve()),'--build',str(a.build.resolve()),'--method',method,'--direction',direction,'--output',str(a.output.resolve()/(method+'-'+direction))]
            if a.model:cmd+=['--model',str(a.model.resolve())]
            if a.sanity:cmd.append('--sanity')
            write(a.output/(method+'-'+direction+'-command.json'),dict(argv=cmd,environment={k:v for k,v in environment(a.build).items() if k.startswith(('LEANSTREAM_','TRITON_','PYTHON'))}))
            with (a.output/(method+'-'+direction+'.log')).open('w') as f:subprocess.run(cmd,env=environment(a.build),stdout=f,stderr=subprocess.STDOUT,check=True)
    if not a.sanity:compare(a.output)
if __name__=='__main__':main()
