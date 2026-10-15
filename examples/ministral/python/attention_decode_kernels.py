"""Decode-only kernels: exact HF half-rounded RoPE and direct GQA."""
import triton
import triton.language as tl

@triton.jit
def rope_decode(Q,K,C,S,P,OQ,OK,D:tl.constexpr=128,POSITIONED:tl.constexpr=False):
    h=tl.program_id(0);d=tl.arange(0,D);other=(d+D//2)%D
    pos=0 if POSITIONED else tl.load(P)
    c=tl.load(C+pos*D+d).to(tl.float32);s=tl.load(S+pos*D+d).to(tl.float32)
    if h<32:
        x=tl.load(Q+h*D+d).to(tl.float32);r=tl.load(Q+h*D+other).to(tl.float32)
    else:
        x=tl.load(K+(h-32)*D+d).to(tl.float32);r=tl.load(K+(h-32)*D+other).to(tl.float32)
    r=tl.where(d<D//2,-r,r)
    # HF creates two FP16 products before its FP16 add. Preserve those roundings.
    a=(x*c).to(tl.float16).to(tl.float32);b=(r*s).to(tl.float16).to(tl.float32)
    y=(a+b).to(tl.float16)
    if h<32:tl.store(OQ+h*D+d,y)
    else:tl.store(OK+(h-32)*D+d,y)

@triton.jit(do_not_specialize=['N','CHECK_TILES','EPOCH'])
def gqa_decode(Q,K,V,O,N,Mid=None,Meta=None,EPOCH=0,CHECK_TILES=0,Bell=None,
               CHECKPOINT:tl.constexpr=False,NOTIFY:tl.constexpr=False,HEAD_START:tl.constexpr=0,
               PREFIX_ONLY:tl.constexpr=False,
               D:tl.constexpr=128,BLOCK:tl.constexpr=128):
    h=tl.program_id(0)+HEAD_START;kh=h//4;d=tl.arange(0,D);t=tl.arange(0,BLOCK)
    q=tl.load(Q+h*D+d).to(tl.float32)
    maximum=tl.full((),float('-inf'),tl.float32);denom=tl.full((),0,tl.float32);acc=tl.full((D,),0,tl.float32)
    for start in range(tl.cdiv(N,BLOCK)):
        idx=start*BLOCK+t
        k=tl.load(K+kh*N*D+idx[:,None]*D+d[None,:],idx[:,None]<N,0).to(tl.float32)
        score=tl.sum(k*q[None,:],1)*0.08838834764831845
        score=tl.where(idx<N,score,float('-inf'))
        m=tl.maximum(maximum,tl.max(score,0));alpha=tl.exp(maximum-m);prob=tl.exp(score-m)
        v=tl.load(V+kh*N*D+idx[:,None]*D+d[None,:],idx[:,None]<N,0).to(tl.float32)
        acc=acc*alpha+tl.sum(prob[:,None]*v,0);denom=denom*alpha+tl.sum(prob,0);maximum=m
        if CHECKPOINT:
            if start+1==CHECK_TILES:
                # One 128-d payload per head; no CTA waits for any other head.
                tl.store(Mid+h*D+d,(acc/denom).to(tl.float16))
                token=tl.inline_asm_elementwise(
                    'fence.acq_rel.sys; mov.u32 $0, 0;',constraints='=r,r,~{memory}',
                    args=[d],dtype=tl.int32,is_pure=False,pack=1)
                tl.debug_barrier()
                arrived=tl.atomic_add(Meta+1,1+tl.sum(token,0),sem='acq_rel',scope='sys')
                if arrived==31:
                    tl.atomic_xchg(Meta,EPOCH,sem='release',scope='sys')
                    if NOTIFY:tl.atomic_add(Bell,1,sem='release',scope='sys')
        if PREFIX_ONLY:
            if start+1==CHECK_TILES:
                tl.store(Mid+h*D+d,(acc/denom).to(tl.float16))
    tl.store(O+h*D+d,acc/denom)

@triton.jit
def gqa_decode_splitkv2(Q,K,V,S,N,HEAD_START:tl.constexpr=0,D:tl.constexpr=128,BLOCK:tl.constexpr=128):
    h=tl.program_id(0)+HEAD_START;part=tl.program_id(1);kh=h//4
    d=tl.arange(0,D);t=tl.arange(0,BLOCK)
    q=tl.load(Q+h*D+d).to(tl.float32)
    half=tl.cdiv(N,2)
    begin=part*half;end=tl.minimum(begin+half,N)
    maximum=tl.full((),float('-inf'),tl.float32);denom=tl.full((),0,tl.float32)
    acc=tl.full((D,),0,tl.float32)
    for tile in range(tl.cdiv(half,BLOCK)):
        idx=begin+tile*BLOCK+t;valid=idx<end
        k=tl.load(K+kh*N*D+idx[:,None]*D+d[None,:],valid[:,None],0).to(tl.float32)
        score=tl.sum(k*q[None,:],1)*0.08838834764831845
        score=tl.where(valid,score,float('-inf'))
        m=tl.maximum(maximum,tl.max(score,0));alpha=tl.exp(maximum-m);prob=tl.exp(score-m)
        v=tl.load(V+kh*N*D+idx[:,None]*D+d[None,:],valid[:,None],0).to(tl.float32)
        acc=acc*alpha+tl.sum(prob[:,None]*v,0);denom=denom*alpha+tl.sum(prob,0);maximum=m
    base=S+(h*2+part)*(D+2)
    tl.store(base,maximum);tl.store(base+1,denom);tl.store(base+2+d,acc)

@triton.jit
def gqa_decode_splitkv2_merge(S,O,HEAD_START:tl.constexpr=0,D:tl.constexpr=128):
    h=tl.program_id(0)+HEAD_START;d=tl.arange(0,D)
    a=S+(h*2)*(D+2);b=a+D+2
    m0=tl.load(a);m1=tl.load(b);l0=tl.load(a+1);l1=tl.load(b+1)
    m=tl.maximum(m0,m1);w0=tl.exp(m0-m);w1=tl.exp(m1-m)
    z0=tl.load(a+2+d);z1=tl.load(b+2+d)
    tl.store(O+h*D+d,(w0*z0+w1*z1)/(w0*l0+w1*l1))

@triton.jit
def partial_output(Z,W,O,COLS:tl.constexpr,START:tl.constexpr,TILE:tl.constexpr=256,ROWS:tl.constexpr=16,
                   W_STRIDE:tl.constexpr=0,W_START:tl.constexpr=0):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS);c=tl.arange(0,TILE)
    acc=tl.full((ROWS,TILE),0,tl.float32)
    for start in range(0,COLS,TILE):
        x=tl.load(Z+START+start+c).to(tl.float32)
        w=tl.load(W+row[:,None]*(W_STRIDE if W_STRIDE else COLS)+W_START+start+c[None,:]).to(tl.float32)
        acc+=w*x[None,:]
    tl.store(O+row,tl.sum(acc,1))

@triton.jit
def add_output(A,B,Y,BLOCK:tl.constexpr=256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i,tl.load(A+i)+tl.load(B+i))
