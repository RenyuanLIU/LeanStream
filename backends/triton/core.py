"""Fixed standalone tiling adapted only for real cache/staging descriptors and h.
Descriptor: logical group, physical scale group, source slot, destination slot,
source kind (0=device packed cache, 1=mapped packed staging).
"""
import triton
import triton.language as tl
from .primitives import _finish, _clock

@triton.jit
def weight(Cache,Staging,Scales,slot,physical,source,c,valid,PROJ:tl.constexpr,WIDTH:tl.constexpr=4096):
    offset=slot[:,None]*(3*WIDTH//2)+PROJ*(WIDTH//2)+c[None,:]//2
    a=tl.load(Cache+offset,valid[:,None] & (source[:,None]==0),0).to(tl.int32)
    b=tl.load(Staging+offset,valid[:,None] & (source[:,None]==1),0).to(tl.int32)
    q=((a+b)>>((c[None,:]%2)*4))&15
    q=tl.where(q>=8,q-16,q)
    scale=tl.load(Scales+physical[:,None]*(3*WIDTH//128)+PROJ*(WIDTH//128)+c[None,:]//128,valid[:,None],0).to(tl.float32)
    return (q.to(tl.float32)*scale).to(tl.float16).to(tl.float32)

@triton.jit(do_not_specialize=['EPOCH'])
def runtime_split(X,Cache,Staging,Scales,Desc,H,Partial,Ready,Started,Done,Stamp,EPOCH,
                  N:tl.constexpr,OFFSET:tl.constexpr,WAIT:tl.constexpr,G:tl.constexpr=2,
                  WIDTH:tl.constexpr=4096,Count=None,DYNAMIC:tl.constexpr=False):
    # Selected standalone configuration: ROWS=4, TILE=256, warps=4, stages=1.
    program=tl.program_id(0);p=program+OFFSET
    tl.store(Stamp+p*4,_clock());tl.atomic_xchg(Started+p,EPOCH,sem='release',scope='sys')
    if WAIT:
        ready=tl.atomic_add(Ready,0,sem='acquire',scope='sys')
        while ready!=EPOCH:ready=tl.atomic_add(Ready,0,sem='acquire',scope='sys')
    tl.store(Stamp+p*4+1,_clock())
    # The control value is a count of logical groups; this kernel indexes
    # individual neurons. Keep the unit conversion after the READY acquire.
    limit=tl.load(Count)*G if DYNAMIC else N
    if not DYNAMIC or program*4<limit:
        n=program*4+tl.arange(0,4);valid=n<limit
        physical=tl.load(Desc+n//G*5+1,valid,0)*G+n%G
        slot=tl.load(Desc+n//G*5+2,valid,0)*G+n%G
        dest=tl.load(Desc+n//G*5+3,valid,0)*G+n%G
        source=tl.load(Desc+n//G*5+4,valid,0)
        compute_clock=tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',
            constraints='=l,r,~{memory}',args=[tl.sum(physical+slot+dest+source,0)],
            dtype=tl.uint64,is_pure=False,pack=1)
        tl.store(Stamp+p*4+2,compute_clock)
        lane=tl.arange(0,256)
        gate=tl.full((4,256),0,tl.float32);up=tl.full((4,256),0,tl.float32)
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
            tl.store(Partial+p*WIDTH+c,tl.sum(h[:,None]*down,0))
    else:
        # Unissued programs have no logical group. The reduction still includes
        # the fixed maximum grid, so its inactive partials must be zero.
        tl.store(Stamp+p*4+2,_clock())
        lane=tl.arange(0,256)
        for base in range(0,WIDTH,256):
            tl.store(Partial+p*WIDTH+base+lane,0.)
    tl.store(Stamp+p*4+3,_clock());_finish(Done,EPOCH,p,lane)
