"""Model-independent production-kernel numerics and pre-enqueued publication."""
import argparse
import sys
from pathlib import Path
import time
import math
import numpy as np
import torch
import triton
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backends.triton.core import runtime_split
from backends.triton.persistent import persistent_split
from backends.triton.reduction import reduce_split
from backends.triton.binding import NativeBackend
from backends.triton.mapped import Mapped,library

p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
lib=library(a.library)
H,G,CORE,PB,TOTAL=256,2,2,32,4
PA=(CORE*G+3)//4;PROGRAMS=PA+PB
mapped=[]
def alloc(n,dtype):
    value=Mapped(lib,n,dtype);mapped.append(value);return value
core=alloc(CORE*5,np.int32);desc=alloc(PB*5,np.int32)
indices=alloc(PB,np.int32);control=alloc(32,np.int32)
started=alloc(PROGRAMS,np.int32);done=alloc(PROGRAMS,np.int32)
stamp=alloc(PROGRAMS*4,np.int64);finished=alloc(H//32,np.int32)
counts=alloc(PB,np.int32)
N=CORE+PB
for i in range(CORE):core.host.reshape(-1,5)[i]=[i,i,i,i,0]
for i in range(PB):desc.host.reshape(-1,5)[i]=[CORE+i,CORE+i,CORE+i,CORE+i,i%2]
indices.host[:TOTAL]=np.arange(TOTAL)
control.host.fill(0);control.host[:4]=[TOTAL,0,TOTAL,1]
x=torch.full((H,),.1,device='cuda',dtype=torch.float16)
cache=torch.full((N*G*3*H//2,),0x11,device='cuda',dtype=torch.uint8)
staging=cache.clone();scales=torch.full((N*G*3*H//128,),.05,device='cuda',dtype=torch.float16)
h=torch.empty(N*G,device='cuda',dtype=torch.float16)
partial=torch.empty((PROGRAMS,H),device='cuda',dtype=torch.float32)
y=torch.empty(H,device='cuda',dtype=torch.float16);claim=torch.zeros(3,device='cuda',dtype=torch.int32)
# Compile and warm each pre-existing specialization before native binding.
ka=runtime_split[(PA,)](x,cache,staging,scales,core.tensor,h,partial,control.tensor,
    started.tensor,done.tensor,stamp.tensor,1,CORE*G,0,False,G,H,num_warps=4,num_stages=1)
ks=[]
for i,ctas in enumerate([4,8,16,32]):
    ks.append(persistent_split[(ctas,)](x,cache,staging,scales,desc.tensor,h,partial,
        indices.tensor,control.tensor,claim,done.tensor,stamp.tensor,1,i+1,PA,PB,ctas,G,H,
        Counts=counts.tensor,CHECK=True,num_warps=4,num_stages=1))
kr=reduce_split[(H//32,)](partial,y,finished.tensor,1,PROGRAMS,
    triton.next_power_of_2(PA+(PB+1)//2),32,WIDTH=H,CORE=PA,num_warps=4,num_stages=1)
torch.cuda.synchronize()
# This table is explicit synthetic correctness data, never performance tuning.
backend=NativeBackend(a.library,offline_table=[[0,1,2,3]])
common=dict(X=x,Cache=cache,Staging=staging,Scales=scales,H=h,Partial=partial,
    Ready=control.tensor,Started=started.tensor,Done=done.tensor,Stamp=stamp.tensor,
    EPOCH=1,Indices=indices.tensor,Control=control.tensor,Claim=claim,CLAIM_EPOCH=1,Counts=counts.tensor)
backend.bind(3,0,ka,dict(common,Desc=core.tensor),PA)
for i,k in enumerate(ks):backend.bind(i,1,k,dict(common,Desc=desc.tensor),[4,8,16,32][i])
backend.bind(3,2,kr,dict(Partial=partial,Y=y,Done=finished.tensor,EPOCH=1,HIDDEN_EPOCH=0),H//32)
backend.seed_claim(4)
for i,ctas in enumerate([4,8,16,32]):
    epoch=i+2;counts.host.fill(0);control.host.fill(0);control.host[2]=TOTAL
    backend.enqueue(0,i,epoch,torch.cuda.current_stream().cuda_stream)
    # No callback launches B: it is already submitted and waits for publication.
    lib.tm_release(control.context,3,epoch)
    lib.tm_release(control.context,0,2)
    deadline=time.monotonic()+5
    while int(counts.host[:TOTAL].sum())!=2:
        if time.monotonic()>deadline:raise AssertionError('First ready prefix did not execute')
        time.sleep(.001)
    lib.tm_release(control.context,0,TOTAL)
    torch.cuda.synchronize()
    assert np.all(counts.host[:TOTAL]==1) and np.all(counts.host[TOTAL:]==0)
    assert np.all(finished.host==epoch)
    half=lambda z:float(np.float16(z))
    gate=half(H*half(.1)*half(.05));silu=half(gate/(1+math.exp(-gate)))
    activation=half(silu*gate);expected=half((CORE+TOTAL)*G*activation*half(.05))
    torch.testing.assert_close(y,torch.full_like(y,expected),rtol=0,atol=.002)
    print(f'PASS: Triton B={ctas} CTAs, LUT-selected native enqueue, cache/staging, exactly-once ready prefix, numerical reduction')
backend.close()
for value in reversed(mapped):value.close()
