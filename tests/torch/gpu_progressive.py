"""Controlled producer gate: concurrency correctness only, never a benchmark."""
import argparse,ctypes as C,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from examples.custom_pytorch.run import ToyModel,pack,predictor
from leanstream.integrations.torch import *
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec,_=pack(a.output);model=ToyModel().cuda().half().eval();x=torch.full((1,1,512),.1,device='cuda',dtype=torch.float16)
state={};handle=None

def final(stage,layer,activation,budget):
    lib=state['lib'];native=state['native']
    assert state['attention_completed_while_io_held']
    assert lib.lt_test_inflight(native.context,6)==1
    handle.refine(0,[7,5,2],2)
    # Started obsolete group6 remains inflight after a newer priority generation.
    assert lib.lt_test_inflight(native.context,6)==1
    lib.lt_test_hold(0)
    return predictor(stage,layer,activation,budget)

schedule=PredictionSchedule(final=PredictionStage(final,5,5),early=PredictionStage(predictor,4,4))
with torch.inference_mode():
    with attach_leanstream(model,points=ModelPoints('layers'),weights=[spec],compute=ComputeSpec(a.library,check_execution=True),cache_policy=[CachePolicy((0,1,2),(0,1))],prediction=schedule) as handle:
        r=handle.entries[0][3];native=r.bundle.native;lib=native.lib;state.update(lib=lib,native=native)
        lib.lt_test_hold.argtypes=[C.c_int];lib.lt_test_hold.restype=None
        for name in ['lt_test_entered','lt_test_bytes']:getattr(lib,name).restype=C.c_uint64
        lib.lt_test_inflight.argtypes=[C.c_void_p,C.c_int];lib.lt_test_inflight.restype=C.c_int
        def before(module,args):
            deadline=time.monotonic()+5
            while lib.lt_test_entered()<4:
                if time.monotonic()>deadline:raise AssertionError('No started SSD reads')
                time.sleep(.001) # correctness gate only
            assert native.stats()['generation']==1 and lib.lt_test_bytes()==0
        def after(module,args,output):
            torch.cuda.current_stream().synchronize() # correctness proof only, absent in normal integration
            assert lib.lt_test_bytes()==0
            state['attention_completed_while_io_held']=True
        hooks=[model.layers[0].self_attn.register_forward_pre_hook(before),model.layers[0].self_attn.register_forward_hook(after)]
        lib.lt_test_hold(1)
        try:
            y=model(x);handle.drain();assert torch.isfinite(y).all()
            assert native.stats()['generation']==3 and lib.lt_test_bytes()>=5*1536
            assert r.bundle.count_host[:3].tolist()==[1,1,1] and not r.bundle.count_host[3:].any()
            assert r.bundle.desc_host.reshape(-1,5)[:3,0].tolist()==[2,5,7]
        finally:
            lib.lt_test_hold(0)
            for hook in hooks:hook.remove()
print('PASS: early real SSD reads, attention completes while IO held, three generations, non-preemptive inflight, final-only exactly-once READY consumption, retirement')
