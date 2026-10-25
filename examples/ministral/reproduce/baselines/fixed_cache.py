"""Two ungrouped baselines sharing immutable individual-neuron storage/cache."""
import ctypes as C
import hashlib,json
from pathlib import Path
import numpy as np
import torch,triton
import triton.language as tl
from torch import nn
from backends.triton.core import weight
from python.predictor_cpu_native import aligned
from scripts.triton_split_probe import Mapped,library

K=2867;HOT=2437;I=14336;H=4096;PROGRAMS=(K+3)//4
@triton.jit
def sparse(X,Cache,Staging,Scales,Desc,Partial,K:tl.constexpr=2867,H:tl.constexpr=4096):
    p=tl.program_id(0);n=p*4+tl.arange(0,4);valid=n<K
    physical=tl.load(Desc+n*5+1,valid,0);slot=tl.load(Desc+n*5+2,valid,0)
    source=tl.load(Desc+n*5+4,valid,0);lane=tl.arange(0,256)
    gate=tl.full((4,256),0,tl.float32);up=tl.full((4,256),0,tl.float32)
    for base in range(0,H,256):
        c=base+lane;x=tl.load(X+c).to(tl.float32)
        gate+=weight(Cache,Staging,Scales,slot,physical,source,c,valid,0,H)*x[None,:]
        up+=weight(Cache,Staging,Scales,slot,physical,source,c,valid,1,H)*x[None,:]
    g=tl.sum(gate,1).to(tl.float16).to(tl.float32);u=tl.sum(up,1).to(tl.float16).to(tl.float32)
    silu=(g/(1+tl.exp(-g))).to(tl.float16).to(tl.float32);h=(silu*u).to(tl.float16).to(tl.float32)
    for base in range(0,H,256):
        c=base+lane
        down=weight(Cache,Staging,Scales,slot,physical,source,c,valid,2,H)
        tl.store(Partial+p*H+c,tl.sum(h[:,None]*down,0))
@triton.jit
def reduce(Partial,Y,H:tl.constexpr=4096,PROGRAMS:tl.constexpr=717):
    c=tl.program_id(0)*32+tl.arange(0,32);p=tl.arange(0,1024)
    v=tl.load(Partial+p[:,None]*H+c[None,:],p[:,None]<PROGRAMS,0)
    tl.store(Y+c,tl.sum(v,0))

def cache_hash(layers):
    h=hashlib.sha256()
    for m in layers:h.update(m.hot_ids.tobytes())
    return h.hexdigest()
class MLP(nn.Module):
    def __init__(self,original,runtime,index,folder,hot):
        super().__init__();self.original=original;self.runtime=runtime;self.index=index
        self.hot_ids=np.asarray(hot,np.int32).copy();self.hot_ids.flags.writeable=False
        self.hidden=Mapped(runtime.mapped_lib,H,np.float16)
        self.staging=Mapped(runtime.mapped_lib,K*6144,np.uint8)
        self.desc=Mapped(runtime.mapped_lib,K*5,np.int32)
        q=np.memmap(folder/'neurons.bin',mode='r',dtype=np.uint8,shape=(I,6144))
        self.cache=torch.from_numpy(np.array(q[self.hot_ids],copy=True)).cuda().reshape(-1)
        self.scales=torch.from_numpy(np.fromfile(folder/'scales.bin',np.float16)).cuda()
        self.partial=torch.empty((PROGRAMS,H),device='cuda',dtype=torch.float32)
        self.y=torch.empty((1,1,H),device='cuda',dtype=torch.float16)
        dim=runtime.dim
        self.features=aligned(np.fromfile(folder/f'features_{dim}.bin',np.int32))
        self.pred_weights=aligned(np.fromfile(folder/f'predictor_{dim}.bin',np.uint8))
        self.pred_scales=aligned(np.fromfile(folder/f'predictor_{dim}_scales.bin',np.float32))
        runtime.check(runtime.lib.fc_bind(runtime.ctx,index,str(folder/'neurons.bin').encode(),dim,
            self.features.ctypes.data,self.pred_weights.ctypes.data,self.pred_scales.ctypes.data,
            self.hot_ids.ctypes.data,self.hidden.host.ctypes.data,self.staging.host.ctypes.data,self.desc.host.ctypes.data))
    def begin(self,module,inputs):
        if self.runtime.decode and self.runtime.mode=='one_shot':self.runtime.start(self.index,inputs[0])
    def forward(self,x):
        r=self.runtime
        if not r.decode:return self.original(x)
        if r.mode=='blocking':r.start(self.index,x)
        r.check(r.lib.fc_wait(r.ctx)) # Required weights must be ready; no profiling wait.
        counts=(C.c_uint64*5)();r.lib.fc_counts(r.ctx,self.index,counts)
        if r.collect:
            r.counters.append(dict(sample_id=r.sample_id,token_index=r.token_index,layer=self.index,
                selected_neurons=counts[0],fixed_hot_hits=counts[1],SSD_misses=counts[2],
                miss_ratio=counts[2]/K,requested_useful_bytes=counts[2]*6144,
                actual_physical_bytes_read=counts[3],read_request_count=counts[4]))
        if r.sanity:
            d=self.desc.host.reshape(K,5)
            assert len(np.unique(d[:,0]))==K and np.all((d[:,0]>=0)&(d[:,0]<I))
            assert counts[0]==K and counts[3]==counts[2]*6144 and counts[4]==counts[2]
            hotmap={int(n):j for j,n in enumerate(self.hot_ids)}
            for n,physical,slot,dest,source in d:
                assert n==physical and (source==0)==(int(n) in hotmap)
                if source==0:assert slot==hotmap[int(n)]
            r.audit.append(dict(layer=self.index,token_index=r.token_index,ids_sha256=hashlib.sha256(d[:,0].tobytes()).hexdigest(),
                finite=False,hot_hits=int(counts[1]),misses=int(counts[2])))
        sparse[(PROGRAMS,)](x,self.cache,self.staging.tensor,self.scales,self.desc.tensor,self.partial,num_warps=4,num_stages=1)
        reduce[(H//32,)](self.partial,self.y,num_warps=4,num_stages=1)
        if r.sanity:
            # Sanity only, outside performance: same selected indices Python math.
            from pack.int4_format import dequantize
            d=self.desc.host.reshape(K,5);q=np.empty((K,3,2048),np.uint8)
            host_cache=self.cache.cpu().numpy().reshape(HOT,3,2048)
            stage=self.staging.host.reshape(K,3,2048)
            for j,row in enumerate(d):q[j]=stage[row[2]] if row[4] else host_cache[row[2]]
            disk=np.memmap(r.root/f'layer_{self.index:02d}'/'neurons.bin',mode='r',dtype=np.uint8,shape=(I,3,2048))
            np.testing.assert_array_equal(q,disk[d[:,0]])
            scales=self.scales.cpu().numpy().reshape(I,3,32)[d[:,0]]
            w=torch.from_numpy(dequantize(q,scales)).cuda()
            g=torch.nn.functional.linear(x,w[:,0]);u=torch.nn.functional.linear(x,w[:,1])
            expected=(torch.nn.functional.silu(g)*u).float()@w[:,2].float()
            assert torch.isfinite(self.y).all().item()
            torch.testing.assert_close(self.y,expected.half(),rtol=.02,atol=.002)
            r.audit[-1]['finite']=True
        return self.y

class Runtime:
    def __init__(self,model,root,native,mode,sanity=False):
        self.mode=mode;self.dim=1638 if mode=='blocking' else 512
        self.root=Path(root)
        self.decode=False;self.collect=False;self.sanity=sanity;self.counters=[];self.audit=[]
        self.lib=C.CDLL(str(native));p=C.c_void_p;i=C.c_int
        specs={'fc_create':([],p),'fc_bind':([p,i,C.c_char_p,i]+[p]*7,i),
               'fc_start':([p,i,p,p],i),'fc_wait':([p],i),'fc_counts':([p,i,p],i),'fc_destroy':([p],None)}
        for name,(args,ret) in specs.items():f=getattr(self.lib,name);f.argtypes=args;f.restype=ret
        self.ctx=self.lib.fc_create();assert self.ctx
        self.mapped_lib=library(None)
        hot=json.loads((Path(root)/'fixed_hot_cache_coqa_17pct.json').read_text())
        canonical=dict(hot);digest=canonical.pop('artifact_sha256')
        assert hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest
        self.layers=[]
        for index,layer in enumerate(model.model.layers):
            m=MLP(layer.mlp,self,index,Path(root)/f'layer_{index:02d}',hot['per_layer'][index]['neuron_ids'])
            layer.mlp=m;layer.register_forward_pre_hook(m.begin);self.layers.append(m)
        self.hot_hash=cache_hash(self.layers)
    def check(self,rc):
        if rc:raise RuntimeError(f'fixed-cache native status {rc}')
    def start(self,index,x):self.check(self.lib.fc_start(self.ctx,index,x.data_ptr(),torch.cuda.current_stream().cuda_stream))
