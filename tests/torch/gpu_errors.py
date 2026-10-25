"""Bounded real-GPU cancellation and failed-submission retirement tests."""
import argparse
import ctypes as C
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from examples.custom_pytorch.run import ToyModel,pack,predictor
from leanstream.integrations.torch import *
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec,_=pack(a.output);model=ToyModel().cuda().half().eval();x=torch.ones(1,1,512,device='cuda',dtype=torch.float16)*.1
schedule=PredictionSchedule(final=PredictionStage(predictor,5,5),early=PredictionStage(predictor,4,4))
with torch.inference_mode():
    with attach_leanstream(model,points=ModelPoints('layers'),weights=[spec],compute=ComputeSpec(a.library),cache_policy=[CachePolicy((0,1,2),(0,1))],prediction=schedule) as h:
        r=h.entries[0][3];native=r.bundle.native
        f=native.lib.lt_test_fail_after_submit;f.argtypes=[C.c_void_p];f.restype=None;f(native.context)
        try:model(x)
        except RuntimeError as e:assert 'bridge error' in str(e)
        else:raise AssertionError('Expected failed submission with live A/B/reduction')
        assert not native.active
        # A fresh epoch can execute after cancel/sync/transport retirement.
        y=model(x);h.drain();assert torch.isfinite(y).all()
        h.reset();assert not native.active
    assert isinstance(model.layers[0].mlp,torch.nn.Module)
print('PASS: failure after live submission, GPU cancellation exit, callback retirement, fresh epoch recovery, reset and detach')
