import triton
import triton.language as tl

@triton.jit
def _clock():
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)


@triton.jit
def _finish(Done,epoch,pid,token_input):
    token=tl.inline_asm_elementwise('fence.acq_rel.sys; mov.u32 $0, 0;',constraints='=r,r,~{memory}',
                                   args=[token_input],dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()
    tl.atomic_xchg(Done+pid,epoch+tl.sum(token,0),sem='release',scope='sys')


@triton.jit(do_not_specialize=['EPOCH','HIDDEN_EPOCH'])
def reduce_split(Partial,Y,Done,EPOCH,P:tl.constexpr,RP:tl.constexpr,BLOCK:tl.constexpr,
                 PUBLISH:tl.constexpr=False,Residual=None,Hidden=None,Flag=None,HIDDEN_EPOCH=0,
                 Bell=None,NOTIFY:tl.constexpr=False,WIDTH:tl.constexpr=4096):
    c=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p=tl.arange(0,RP)
    value=tl.sum(tl.load(Partial+p[:,None]*WIDTH+c[None,:],p[:,None]<P,0),0)
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


@triton.jit(do_not_specialize=['EPOCH'])
def publish_next_hidden(Y,Residual,Hidden,Flag,EPOCH,Bell=None,NOTIFY:tl.constexpr=False,WIDTH:tl.constexpr=4096):
    # One CTA on the reduction stream. Every writer fences before the barrier;
    # _finish issues one system-release epoch after all 4096 stores are visible.
    c=tl.arange(0,4096)
    valid=c<WIDTH
    value=(tl.load(Residual+c,valid,0).to(tl.float32)+tl.load(Y+c,valid,0).to(tl.float32)).to(tl.float16)
    tl.store(Hidden+c,value,valid)
    # Fence once per physical thread, not once per owned payload element.
    # Four warps cover all 128 writers; the CTA barrier precedes one release.
    _finish(Flag,EPOCH,0,tl.arange(0,128))
    if NOTIFY:tl.atomic_add(Bell,1,sem='release',scope='sys')

