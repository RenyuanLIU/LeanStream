"""Production arithmetic with one pre-enqueued ready-order persistent consumer.
The packed weights, H destinations and original partial-program layout are retained.
"""
import triton
import triton.language as tl
from .core import weight
from .primitives import _clock, _finish

@triton.jit
def _compute_pair(X,Cache,Staging,Scales,Desc,H,Partial,index,n,valid,slots,valid_slots,
             WIDTH:tl.constexpr,G:tl.constexpr,ROWS:tl.constexpr):
    physical=tl.load(Desc+index*5+1,valid,0)*G+n%G
    slot=tl.load(Desc+index*5+2,valid,0)*G+n%G
    dest=tl.load(Desc+index*5+3,valid,0)*G+n%G
    source=tl.load(Desc+index*5+4,valid,0)
    lane=tl.arange(0,256)
    gate=tl.full((ROWS,256),0,tl.float32);up=tl.full((ROWS,256),0,tl.float32)
    for base in range(0,WIDTH,256):
        c=base+lane;x=tl.load(X+c).to(tl.float32)
        gate+=weight(Cache,Staging,Scales,slot,physical,source,c,valid,0,WIDTH)*x[None,:]
        up+=weight(Cache,Staging,Scales,slot,physical,source,c,valid,1,WIDTH)*x[None,:]
    g=tl.sum(gate,1).to(tl.float16).to(tl.float32)
    u=tl.sum(up,1).to(tl.float16).to(tl.float32)
    silu=(g/(1+tl.exp(-g))).to(tl.float16).to(tl.float32)
    h=(silu*u).to(tl.float16).to(tl.float32)
    tl.store(H+dest,h,valid)
    for base in range(0,WIDTH,256):
        c=base+lane;down=weight(Cache,Staging,Scales,slot,physical,source,c,valid,2,WIDTH)
        value=tl.sum(tl.reshape(h[:,None]*down,(2,2,256)),1)
        tl.store(Partial+slots[:,None]*WIDTH+c[None,:],value,valid_slots[:,None])

@triton.jit(do_not_specialize=['EPOCH','CLAIM_EPOCH'])
def persistent_split(X,Cache,Staging,Scales,Desc,H,Partial,Indices,Control,Claim,Done,Stamp,EPOCH,CLAIM_EPOCH,
                     OFFSET:tl.constexpr,PB:tl.constexpr,WORKERS:tl.constexpr,G:tl.constexpr=2,WIDTH:tl.constexpr=4096,Counts=None,CHECK:tl.constexpr=False):
    tl.static_assert(G==2)
    worker=tl.program_id(0);lane=tl.arange(0,256)
    tl.store(Stamp+(OFFSET+worker)*4,_clock())
    initialized=tl.atomic_add(Claim+1,0,sem='acquire',scope='gpu')
    while initialized!=CLAIM_EPOCH:
        if initialized>=0:
            old=tl.atomic_cas(Claim+1,initialized,~CLAIM_EPOCH,sem='acq_rel',scope='gpu')
            if old==initialized:
                tl.store(Claim,0);tl.store(Claim+2,0)
                tl.atomic_xchg(Claim+1,CLAIM_EPOCH,sem='release',scope='gpu')
        initialized=tl.atomic_add(Claim+1,0,sem='acquire',scope='gpu')
    epoch=tl.atomic_add(Control+3,0,sem='acquire',scope='sys')
    error=tl.atomic_add(Control+5,0,sem='acquire',scope='sys')
    while (epoch!=EPOCH)&(error==0):
        epoch=tl.atomic_add(Control+3,0,sem='acquire',scope='sys')
        error=tl.atomic_add(Control+5,0,sem='acquire',scope='sys')
    total=tl.load(Control+2)
    ready=tl.atomic_add(Control,0,sem='acquire',scope='sys')
    choice=tl.where(ready==total,2,1)
    tl.atomic_cas(Claim+2,0,choice,sem='acq_rel',scope='gpu')
    fast=tl.atomic_add(Claim+2,0,sem='acquire',scope='gpu')==2
    tl.store(Stamp+(OFFSET+worker)*4+1,_clock())
    # Every valid group owns one partial; no atomics or global initialization
    # barrier. Only the unused suffix needs zeroing for fixed-size reduction.
    for idx in range(total+worker,PB,WORKERS):
        for base in range(0,WIDTH,256):tl.store(Partial+(OFFSET+idx)*WIDTH+base+lane,0.)
    first=True
    tl.store(Stamp+(OFFSET+worker)*4+2,_clock())
    if fast:
        for start in range(2*worker,total,2*WORKERS):
            if first:tl.store(Stamp+(OFFSET+worker)*4+2,_clock());first=False
            n=tl.arange(0,4);index=start+n//G;valid=index<total
            slots=start+tl.arange(0,2)
            _compute_pair(X,Cache,Staging,Scales,Desc,H,Partial,index,n,valid,OFFSET+slots,slots<total,WIDTH,G,4)
            if CHECK:tl.atomic_add(Counts+slots,1,mask=slots<total,sem='relaxed',scope='sys')
    else:
        active=True
        while active:
            ready=tl.atomic_add(Control,0,sem='acquire',scope='sys')
            head=tl.atomic_add(Claim,0,sem='relaxed',scope='gpu')
            if head>=total:active=False
            elif head<ready:
                end=tl.minimum(head+2,ready)
                old=tl.atomic_cas(Claim,head,end,sem='acq_rel',scope='gpu')
                if old==head:
                    ids=tl.load(Indices+head+tl.arange(0,2),head+tl.arange(0,2)<end,0)
                    if first:tl.store(Stamp+(OFFSET+worker)*4+2,_clock());first=False
                    n=tl.arange(0,4)
                    idx0=tl.sum(tl.where(tl.arange(0,2)==0,ids,0),0)
                    idx1=tl.sum(tl.where(tl.arange(0,2)==1,ids,0),0)
                    index=tl.where(n<G,idx0,idx1);valid=(n<G)|(head+1<end)
                    _compute_pair(X,Cache,Staging,Scales,Desc,H,Partial,index,n,valid,OFFSET+ids,head+tl.arange(0,2)<end,WIDTH,G,4)
                    if CHECK:tl.atomic_add(Counts+ids,1,mask=head+tl.arange(0,2)<end,sem='relaxed',scope='sys')
            else:
                error=tl.atomic_add(Control+5,0,sem='acquire',scope='sys')
                if error!=0:active=False
    tl.store(Stamp+(OFFSET+worker)*4+3,_clock())
    _finish(Done,EPOCH,OFFSET+worker,lane)
