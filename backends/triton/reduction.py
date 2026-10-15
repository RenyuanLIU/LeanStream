import triton
import triton.language as tl
from .primitives import _finish

@triton.jit(do_not_specialize=['EPOCH','HIDDEN_EPOCH'])
def reduce_split(Partial,Y,Done,EPOCH,P:tl.constexpr,RP:tl.constexpr,BLOCK:tl.constexpr,
                 PUBLISH:tl.constexpr=False,Residual=None,Hidden=None,Flag=None,HIDDEN_EPOCH=0,
                 Bell=None,NOTIFY:tl.constexpr=False,WIDTH:tl.constexpr=4096,CORE:tl.constexpr=0):
    c=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p=tl.arange(0,RP)
    physical=tl.where(p<CORE,p,CORE+2*(p-CORE))
    values=tl.load(Partial+physical[:,None]*WIDTH+c[None,:],physical[:,None]<P,0)
    values+=tl.load(Partial+(physical[:,None]+1)*WIDTH+c[None,:],(p[:,None]>=CORE)&(physical[:,None]+1<P),0)
    value=tl.sum(values,0)
    rounded=value.to(tl.float16)
    tl.store(Y+c,rounded) # Identical normal device MLP output.
    if PUBLISH:
        # Match HF half residual + half MLP output: FP32 add, then FP16 rounding.
        hidden=(tl.load(Residual+c).to(tl.float32)+rounded.to(tl.float32)).to(tl.float16)
        tl.store(Hidden+c,hidden)
    _finish(Done,EPOCH,tl.program_id(0),c)
    if PUBLISH:
        # _finish fences every lane before this CTA participates. Acq_rel RMWs
        # chain all 128 CTA payload writes; only the last publishes one epoch.
        arrived=tl.atomic_add(Flag+1,1,sem='acq_rel',scope='sys')
        if arrived==WIDTH//BLOCK-1:
            tl.atomic_xchg(Flag,HIDDEN_EPOCH,sem='release',scope='sys')
            if NOTIFY:tl.atomic_add(Bell,1,sem='release',scope='sys')

