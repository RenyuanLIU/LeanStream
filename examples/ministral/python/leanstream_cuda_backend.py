"""Cold bind already-validated Triton CUDA functions to native compute enqueue.
No model math, prediction, descriptor or stream policy is changed here.
"""
import ctypes as C
import hashlib

def _bind_runtime(runtime):
    if runtime.producer_publication:raise ValueError('Native launch adapter currently requires default input publication')
    manifests=[]
    for m in runtime.layers:
        e=m.engine;n=m.native
        saved=e.epoch;e.epoch=7 # generic non-specialized reduction scalar
        e._public_reduce(m);e.epoch=saved
        common=dict(X=n.rms,Cache=m.packed_cache,Staging=e.staging,Scales=m.reader.device_scales,
                    H=e.h,Partial=e.partial,Ready=e.ready.tensor,Started=e.started.tensor,Done=e.done.tensor,
                    Stamp=e.stamp.tensor,EPOCH=e.epoch,Count=e.ready.tensor[1:2],Indices=e.public_indices.tensor,
                    Control=e.public_control.tensor,Claim=e.public_claim,CLAIM_EPOCH=e.public_launch_epoch,
                    Counts=e.public_exec_counts.tensor if e.public_exec_counts is not None else None)
        maps=[dict(common,Desc=e.a.tensor),dict(common,Desc=e.public_desc.tensor),
              dict(Partial=e.partial,Y=e.y,Done=e.reduced.tensor,EPOCH=e.epoch,HIDDEN_EPOCH=0)]
        kernels=[e.public_a_kernel,e.public_b_kernel,e.public_reduce_kernel]
        e.public_kernel_refs=tuple(kernels)
        f=n.lib.lp_public_cuda_bind;f.restype=C.c_int
        f.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint32,C.c_void_p]
        for stage,(kernel,values) in enumerate(zip(kernels,maps)):
            from backends.triton.binding import pack_kernel, kernel_metadata
            args,epoch,claim,parameters=pack_kernel(kernel,values)
            grid=[e.pa,e.workers if e.public_mode=='progressive' else e.pb,e.width//32][stage]
            n.check(f(n.ctx,stage,kernel.function,grid,kernel.metadata.num_warps,kernel.metadata.shared,
                      args,len(args),epoch,claim,e.public_launch_epoch,e.b_done.cuda_event))
            manifests.append(dict(layer=m.layer,stage=stage,parameters=parameters, **kernel_metadata(kernel,grid)))
        e.native_cuda_bound=True
    return manifests

def bind_runtime(runtime):
    import torch
    try:return _bind_runtime(runtime)
    finally:torch.cuda.current_stream().synchronize() # cold binding only, outside inference timing
