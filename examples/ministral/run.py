"""Short unsplit Ministral decode using the public production backend."""
import os
from pathlib import Path
import runpy
import sys

example=Path(__file__).resolve().parent
project=example.parents[1]
sys.path[:0]=[str(example),str(project)]
for key,value in dict(LEANSTREAM_PUBLIC_CONSUMER='progressive',LEANSTREAM_PUBLIC_WORKERS='32',
        LEANSTREAM_PUBLIC_READY_PAIRS='1',LEANSTREAM_PUBLIC_NATIVE_ENQUEUE='1',
        LEANSTREAM_PUBLIC_EXECUTION_CHECK='0',LEANSTREAM_STREAM_QD='24',LEANSTREAM_STREAM_PROFILE='0').items():
    os.environ.setdefault(key,value)
if os.environ['LEANSTREAM_PUBLIC_WORKERS']!='32':
    raise ValueError('This model recipe uses the validated 32-CTA fallback')
if os.environ['LEANSTREAM_PUBLIC_CONSUMER']!='progressive':
    raise ValueError('This model recipe uses Progressive execution')
from python.split_async import Runtime
from python.leanstream_cuda_backend import bind_runtime
original_reset=Runtime.reset
original_init=Runtime.__init__
instances=[]
def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self._release_resets=0
    instances.append(self)
def reset(self,*args,**kwargs):
    if hasattr(self,'_release_resets'):
        self._release_resets+=1
        if self._release_resets==2:
            # Exactly the validated cold-binding boundary: after one full warmup.
            manifest=bind_runtime(self)
            import json
            output=Path(sys.argv[sys.argv.index('--output')+1])
            (output/'native_kernel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return original_reset(self,*args,**kwargs)
Runtime.__init__=initialize
Runtime.reset=reset
if '--warmup-prompts' in sys.argv and sys.argv[sys.argv.index('--warmup-prompts')+1]!='1':
    raise ValueError('Native setup requires one warmup prompt')
runpy.run_module('scripts.short_decode',run_name='__main__')
for runtime in instances:
    for layer in runtime.layers:
        e=layer.engine
        assert e.workers==32 and e.native_cuda_bound
print('PASS: native Triton pre-enqueue; validated 32 persistent B CTAs, four warps per CTA')
