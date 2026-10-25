"""Cold specialize and bind the existing coordinated Group2 A/B/reduction bundle."""
import importlib
import numpy as np
import torch
from backends.triton.mapped import Mapped
from .native import NativeRuntime

class Group2Bundle:
    def __init__(self,weights,policy,compute):
        import triton
        weights.validate();policy.validate(weights.group_count)
        if not torch.cuda.is_available():raise RuntimeError('CUDA execution required')
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        # tm_alloc performs the authoritative mapping/integrated/native-atomic checks.
        if props.major!=8 or props.minor!=7:raise RuntimeError('This integration is validated only for coherent Jetson SM87')
        self.weights=weights;self.policy=policy;self.compute=compute;self.allocations=[];self.stream=torch.cuda.current_stream().cuda_stream
        self.native=NativeRuntime(weights,policy,compute)
        try:self._prepare(triton)
        except BaseException:
            # Cold setup can have submitted work. Prove completion before releasing.
            torch.cuda.current_stream().synchronize();self.native.abort();self.close();raise
    def _prepare(self,triton):
        w=self.weights;p=self.policy;n=w.group_count;h=w.width;pa=len(p.core)//2;pb=n-len(p.core);programs=pa+pb
        self.x=torch.zeros(h,device='cuda',dtype=torch.float16);self.y=torch.empty_like(self.x)
        self.scales=w.scales.contiguous().flatten().cuda()
        by_id={g.logical_group:g for g in w.layout}
        with open(w.path,'rb') as f:
            resident=[]
            for g in p.resident:f.seek(by_id[g].file_offset);resident.extend(f.read(w.group_bytes))
        self.cache=torch.tensor(resident,dtype=torch.uint8,device='cuda')
        self.activation=torch.empty(n*2,device='cuda',dtype=torch.float16)
        self.partial=torch.empty((programs,h),device='cuda',dtype=torch.float32)
        self.claim=torch.zeros(3,device='cuda',dtype=torch.int32)
        self.staging_host,self.staging=self.native.view(0,n*w.group_bytes,np.uint8)
        self.desc_host,self.desc=self.native.view(2,n*5,np.int32)
        self.index_host,self.indices=self.native.view(3,n,np.int32)
        self.control_host,self.control=self.native.view(4,32,np.int32)
        self.core_host,self.core=self.native.view(5,len(p.core)*5,np.int32)
        self.count_host,self.counts=self.native.view(6,n,np.int32)
        def alloc(count,dtype):
            a=Mapped(self.native.lib,count,dtype);self.allocations.append(a);return a.tensor
        started=alloc(programs,np.int32);done=alloc(max(programs,32),np.int32);stamp=alloc((pa+32)*4,np.int64);reduced=alloc(h//32,np.int32)
        if self.compute.bundle:
            package=importlib.import_module(self.compute.bundle)
            runtime_split,persistent_split,reduce_split=package.runtime_split,package.persistent_split,package.reduce_split
        else:
            from backends.triton.core import runtime_split
            from backends.triton.persistent import persistent_split
            from backends.triton.reduction import reduce_split
        self.native.begin();self.native.finalize(list(p.core))
        # Compile with an empty finalized B plan; never trial-run candidates during decode.
        ka=runtime_split[(pa,)](self.x,self.cache,self.staging,self.scales,self.core,self.activation,self.partial,self.control,started,done,stamp,1,len(p.core)*2,0,False,2,h,num_warps=4,num_stages=1)
        kb=persistent_split[(32,)](self.x,self.cache,self.staging,self.scales,self.desc,self.activation,self.partial,self.indices,self.control,self.claim,done,stamp,1,1,pa,pb,32,2,h,Counts=self.counts,CHECK=self.compute.check_execution,num_warps=4,num_stages=1)
        kr=reduce_split[(h//32,)](self.partial,self.y,reduced,1,programs,triton.next_power_of_2(pa+(pb+1)//2),32,WIDTH=h,CORE=pa,num_warps=4,num_stages=1)
        torch.cuda.current_stream().synchronize();self.native.abort()
        common=dict(X=self.x,Cache=self.cache,Staging=self.staging,Scales=self.scales,H=self.activation,Partial=self.partial,Ready=self.control,Started=started,Done=done,Stamp=stamp,EPOCH=1,Indices=self.indices,Control=self.control,Claim=self.claim,CLAIM_EPOCH=1,Counts=self.counts)
        self.binding=[self.native.bind(0,ka,dict(common,Desc=self.core),pa),self.native.bind(1,kb,dict(common,Desc=self.desc),32),self.native.bind(2,kr,dict(Partial=self.partial,Y=self.y,Done=reduced,EPOCH=1,HIDDEN_EPOCH=0),h//32)]
        self.native.check(self.native.lib.lt_seed(self.native.context,1))
    def enqueue(self,x):
        if torch.cuda.current_stream().cuda_stream!=self.stream:raise ValueError('Bundle is single-stream; drain and reattach to change streams')
        self.x.copy_(x.reshape(-1));self.native.enqueue(self.x,self.y,self.stream)
    def output(self,shape):return self.y.reshape(shape).clone()
    def close(self):
        self.native.close()
        for a in reversed(self.allocations):a.close()
        self.allocations.clear()
