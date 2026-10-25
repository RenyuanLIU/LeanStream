"""Small real-model numerical/decode checks using existing unmodified entry points."""
import argparse,subprocess,sys
from pathlib import Path
if __package__:
    from .common import HERE,EXAMPLE,environment,write
    from .assets import verify
else:
    from common import HERE,EXAMPLE,environment,write
    from assets import verify

def main():
    p=argparse.ArgumentParser()
    for name in ['assets','model','build','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['leanstream','block','one-shot'],default='leanstream');p.add_argument('--numerical',action='store_true');a=p.parse_args()
    verify(a.assets,model_root=a.model);a.output.mkdir(parents=True,exist_ok=False)
    if a.method=='leanstream':
        cmd=[sys.executable,str(EXAMPLE/('numerical.py' if a.numerical else 'run.py')),'--model',str(a.model.resolve()),'--rope',str((a.assets/'metadata/rope.json').resolve()),'--weights',str((a.assets/'leanstream_weights').resolve()),'--predictor-root',str((a.assets/'full_predictors').resolve()),'--p1-lite-root',str((a.assets/'p1_lite_predictors').resolve()),'--predictor-library',str((a.build/'libpredictor.so').resolve()),'--shared',str((a.build/'libmapped.so').resolve()),'--prompts',str((a.assets/'benchmarks/speed_prompts.jsonl').resolve())]
        if not a.numerical:cmd+=['--indices','0','--warmup-prompts','1','--max-decode-tokens','4','--no-layer-events','--output',str((a.output/'decode').resolve())]
    else:
        cmd=[sys.executable,str(HERE/'worker.py'),'--assets',str(a.assets.resolve()),'--model',str(a.model.resolve()),'--build',str(a.build.resolve()),'--method',a.method,'--sanity','--output',str((a.output/'baseline').resolve())]
    write(a.output/'launch_command.json',dict(argv=cmd))
    with (a.output/'smoke.log').open('w') as f:subprocess.run(cmd,env=environment(a.build),stdout=f,stderr=subprocess.STDOUT,check=True)
    print('PASS:',a.method,'existing numerical/decode smoke; see smoke.log')
if __name__=='__main__':main()
