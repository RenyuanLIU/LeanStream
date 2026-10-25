"""Small real Runtime + SSD + Triton example; no downloaded model or predictor."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from leanstream.integrations.torch import (ModelPoints,WeightSpec,GroupLayout,CachePolicy,ComputeSpec,PredictionStage,PredictionSchedule,attach_leanstream,inspect_model)

class GatedMLP(torch.nn.Module):
    def __init__(self,width=512,groups=8):
        super().__init__();self.gate_proj=torch.nn.Linear(width,groups*2,bias=False);self.up_proj=torch.nn.Linear(width,groups*2,bias=False);self.down_proj=torch.nn.Linear(groups*2,width,bias=False)
    def forward(self,x):return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))

class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=torch.nn.Linear(512,512,bias=False);self.post_attention_layernorm=torch.nn.LayerNorm(512);self.mlp=GatedMLP()
    def forward(self,hidden_states):
        x=hidden_states+self.self_attn(hidden_states);return x+self.mlp(self.post_attention_layernorm(x))

class ToyModel(torch.nn.Module):
    def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([ToyLayer()])
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        return x

def pack(directory):
    """Offline synthetic example only: physical order differs from logical IDs."""
    directory.mkdir(parents=True,exist_ok=True);path=directory/'weights.bin'
    if path.exists():raise FileExistsError('Use a new output directory; do not overwrite artifacts')
    torch.manual_seed(11);q=torch.randint(-3,4,(8,2,3,512),dtype=torch.int8);scales=torch.full((8,2,3,4),.05,dtype=torch.float16)
    physical=[3,0,6,1,7,2,5,4];layout=[];blocks=[None]*8
    for logical,p in enumerate(physical):
        row=q[logical].to(torch.int16).reshape(-1);packed=((row[::2]&15)|((row[1::2]&15)<<4)).to(torch.uint8)
        blocks[p]=bytes(packed.tolist());layout.append(GroupLayout(logical,p,p*1536,1536))
    path.write_bytes(b''.join(blocks));return WeightSpec(path,512,tuple(layout),scales),q

def predictor(stage,layer,activation,budget):
    # Metadata-only synthetic predictor: no GPU->CPU transfer, wait or trained weights.
    return ([6,4,3,5] if stage=='layer_input' else [0,1,2,5,7])[:budget]

def reference(x,q,selected,multiplier=1):
    # Independent FP32 dequantization/reductions with explicit reference FP16 rounding.
    scale=float(torch.tensor(.05,dtype=torch.float16))
    weights=(q.cuda().float()*scale).half().float()*multiplier
    gate=(weights[:,:,0]*x.float()).sum(-1).half().float()
    up=(weights[:,:,1]*x.float()).sum(-1).half().float()
    h=(torch.nn.functional.silu(gate).half().float()*up).half().float()
    return (h[selected,:,None]*weights[selected,:,2]).sum((0,1)).half()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['inspect','basic','progressive','validate'],default='inspect');p.add_argument('--library',type=Path);p.add_argument('--output',type=Path);p.add_argument('--bundle');p.add_argument('--weight-multiplier',type=float,default=1);a=p.parse_args()
    torch.manual_seed(42);model=ToyModel().eval()
    points=ModelPoints(layers='layers');report=inspect_model(model,points);print(report)
    if a.mode=='inspect':return
    if a.output is None or a.library is None:p.error('--output and --library required for execution')
    a.output.mkdir(parents=True,exist_ok=False);spec,q=pack(a.output/'assets');model=model.cuda().half();torch.set_num_threads(1)
    policy=CachePolicy((0,1,2),(0,1));x=torch.randn((1,1,512),device='cuda',dtype=torch.float16)*.1
    layer=model.layers[0]
    with torch.inference_mode():
        residual=x+layer.self_attn(x);mlp_input=layer.post_attention_layernorm(residual)
        expected=residual+reference(mlp_input.reshape(-1),q,[0,1,2,5,7],a.weight_multiplier).reshape_as(x)
        methods=['basic','progressive'] if a.mode=='validate' else [a.mode];results=[]
        for method in methods:
            schedule=PredictionSchedule(final=PredictionStage(predictor,5,5),early=PredictionStage(predictor,4,4) if method=='progressive' else None)
            original=layer.mlp;attention=layer.self_attn
            handle=attach_leanstream(model,points=points,weights=[spec],compute=ComputeSpec(a.library,a.bundle,True),cache_policy=[policy],prediction=schedule,mode=method)
            try:
                # Dense prefill remains the original path; no low-memory claim.
                prefill=torch.randn((1,3,512),device='cuda',dtype=torch.float16)
                dense=original(layer.post_attention_layernorm(prefill+layer.self_attn(prefill)))
                actual=model(prefill);torch.testing.assert_close(actual,prefill+layer.self_attn(prefill)+dense)
                wrapped=handle.entries[0][3]
                y=model(x);handle.drain();torch.cuda.synchronize()
                torch.testing.assert_close(y,expected,rtol=0,atol=.002)
                native=wrapped.bundle.native;stats=native.stats();control=wrapped.bundle.control_host.copy();descs=wrapped.bundle.desc_host.reshape(-1,5)[:3].copy();counts=wrapped.bundle.count_host.copy()
                assert stats['requests']>=2 and stats['generation']>= (2 if method=='progressive' else 1)
                assert control[0]==control[2]==3 and sorted(wrapped.bundle.index_host[:3].tolist())==[0,1,2]
                assert counts[:3].tolist()==[1,1,1] and not counts[3:].any(),counts
                assert set(descs[:,4].tolist())=={0,1};assert descs[:,0].tolist()==[2,5,7]
                # Reuse requires previous GPU+callback retirement; no stale epoch/double claim.
                first_epoch=int(control[3]);torch.testing.assert_close(model(x),expected,rtol=0,atol=.002);handle.drain()
                assert int(wrapped.bundle.control_host[3])==first_epoch+1
                assert layer.self_attn is attention
                results.append(dict(mode=method,status='PASS',max_abs_error=float((y-expected).abs().max()),native_stats=stats,control=control.tolist(),descriptors=descs.tolist(),counts=counts.tolist(),binding=wrapped.bundle.binding,trace=wrapped.trace))
            finally:handle.close()
            assert layer.mlp is original and not layer._forward_pre_hooks and not layer._forward_hooks
        origins={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith(('leanstream.','backends.triton.')) and getattr(module,'__file__',None)}
        if any(ROOT not in Path(path).parents for path in origins.values()):raise RuntimeError('Unexpected framework module origin')
        (a.output/'provenance.json').write_text(json.dumps(dict(native_library=str(a.library.resolve()),native_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),runtime_modules=origins,weight_sha256=hashlib.sha256(spec.path.read_bytes()).hexdigest()),indent=2)+'\n')
        (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
