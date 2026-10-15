"""Two-token, all-layer validation using the existing model numerical checks."""
import argparse
import json
import os
from pathlib import Path
import sys
example=Path(__file__).resolve().parent
sys.path[:0]=[str(example),str(example.parents[1])]
for key,value in dict(LEANSTREAM_PUBLIC_CONSUMER='progressive',LEANSTREAM_PUBLIC_WORKERS='32',
    LEANSTREAM_PUBLIC_READY_PAIRS='1',LEANSTREAM_PUBLIC_NATIVE_ENQUEUE='1',
    LEANSTREAM_PUBLIC_EXECUTION_CHECK='1',LEANSTREAM_PUBLIC_VALIDATE='1',
    LEANSTREAM_STREAM_QD='24',LEANSTREAM_STREAM_PROFILE='0').items():os.environ[key]=value
import torch
from python.ministral3_text_loader import load_text
from python.split_async import Runtime
from python.attention_refine import DirectGQAAttention
from python.leanstream_cuda_backend import bind_runtime
p=argparse.ArgumentParser()
for name in ['model','weights','predictor-root','predictor-library','shared','rope','prompts','p1-lite-root']:
    p.add_argument('--'+name,required=True)
a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(42)
model=load_text(a.model,a.rope)
r=Runtime(model,a.weights,a.predictor_library,a.shared,
    model_spec=dict(H=4096,I=14336,layers=34,active=1434,keep=1219),
    predictor_root=a.predictor_root,predictor_dim=1664,final_qd=24,
    spec_p1_size=1024,ready_scheduler=True,ready_notification='doorbell',p0_budget=32,
    p1_lite_root=a.p1_lite_root,p1_lite_features=512,p1_lite_budget=128,
    no_waste_policy='knee',diagnostic=False,validate=True)
r.reference_all_layers=True
attention=DirectGQAAttention(model,runtime=r)
prompt=json.loads(Path(a.prompts).read_text().splitlines()[0])
def run():
    r.reset()
    y=model(input_ids=torch.tensor([prompt['input_ids']],device='cuda'),use_cache=True,return_dict=True)
    for _ in range(2):
        token=int(y.logits[:,-1].argmax(-1).item())
        y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=y.past_key_values,use_cache=True,return_dict=True)
        r.drain() # Correctness-only descriptor replay; never a decode timing boundary.
    r.drain()
with torch.inference_mode():
    run()
    manifest=bind_runtime(r)
    assert all(row['grid']==[610,32,128][row['stage']] for row in manifest)
    assert all(row['warps_per_cta']==4 for row in manifest)
    run()
    for m in r.layers:
        e=m.engine;count=int(e.public_control.host[2])
        assert (e.public_exec_counts.host[:count]==1).all()
        assert (e.public_exec_counts.host[count:]==0).all()
print('PASS: short all-34-layer model numerical checks, original tolerances, exact descriptors, native A610/B32/reduction128, four warps per CTA')
