"""Resume-safe public Progressive fixed-500 generation and historical scoring."""
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,help='Separate pinned upstream checkpoint directory');p.add_argument('--assets',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',type=int,choices=range(5,11));a=p.parse_args()
    verify(a.assets,model_root=a.model);a.output.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(HERE/'worker.py'),'--assets',str(a.assets.resolve()),'--build',str(a.build.resolve()),'--output',str(a.output.resolve()),'--method','leanstream','--accuracy','--limit',str(a.dry_run or 500)]
    if a.model:cmd+=['--model',str(a.model.resolve())]
    write(a.output/'launch_command.json',dict(argv=cmd,environment={k:v for k,v in environment(a.build).items() if k.startswith(('LEANSTREAM_','TRITON_','PYTHON'))}))
    with (a.output/'generation.log').open('a') as f:subprocess.run(cmd,env=environment(a.build),stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
