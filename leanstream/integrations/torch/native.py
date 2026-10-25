"""Owned runtime handle; every borrowed GPU/host buffer lives until retirement."""
import ctypes as C
from pathlib import Path
import numpy as np
from backends.triton.mapped import library
from backends.triton.binding import pack_kernel,kernel_metadata

class Layout(C.Structure):
    _fields_=[('logical_group',C.c_uint32),('physical_group',C.c_uint32),('file_offset',C.c_uint64),('bytes',C.c_uint32)]

def integers(ids):return (C.c_int32*len(ids))(*ids)

class NativeRuntime:
    def __init__(self,spec,policy,compute):
        self.lib=library(compute.library);self.context=None;self.refs=[];self.active=False
        p=C.c_void_p;u=C.c_uint32;i=C.c_int
        signatures={
            'lt_create':([C.c_char_p,u,u,p,p,u,p,u,u],p),
            'lt_buffer':([p,i],p),'lt_bind':([p,u,u,p,u,u,u,p,i,i,i],i),'lt_seed':([p,u],i),
            'lt_update':([p,p,u,u],i),'lt_enqueue':([p,p,p,p],i),'lt_finalize':([p,p,u],i),
            'lt_wait_io':([p,i],i),'lt_requests':([p],C.c_uint64),'lt_generation':([p],C.c_uint64)}
        for name in ['lt_begin','lt_commit','lt_abort','lt_drain','lt_close']:signatures[name]=([p],i)
        for name,(args,ret) in signatures.items():f=getattr(self.lib,name);f.argtypes=args;f.restype=ret
        layouts=(Layout*spec.group_count)(*(Layout(x.logical_group,x.physical_group,x.file_offset,x.bytes) for x in spec.layout))
        self.context=self.lib.lt_create(str(Path(spec.path).resolve()).encode(),spec.group_count,spec.group_bytes,layouts,integers(policy.resident),len(policy.resident),integers(policy.core),len(policy.core),24)
        if not self.context:raise RuntimeError('Native allocation/O_DIRECT open failed; coherent integrated GPU with host-native atomics required')
    def check(self,rc):
        if rc:raise RuntimeError(f'LeanStream bridge error {rc}')
    def view(self,index,count,dtype):
        import torch
        handle=self.lib.lt_buffer(self.context,index);dtype=np.dtype(dtype)
        host=np.ctypeslib.as_array((C.c_uint8*(count*dtype.itemsize)).from_address(self.lib.tm_host(handle))).view(dtype)
        capsule=C.pythonapi.PyCapsule_New;capsule.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];capsule.restype=C.py_object
        tensor=torch.utils.dlpack.from_dlpack(capsule(self.lib.tm_dlpack(handle,0,dtype.itemsize*8,count),b'dltensor',None))
        self.refs.append(tensor);return host,tensor
    def bind(self,stage,kernel,values,grid):
        args,epoch,claim,names=pack_kernel(kernel,values)
        self.check(self.lib.lt_bind(self.context,3,stage,kernel.function,grid,kernel.metadata.num_warps,kernel.metadata.shared,args,len(args),epoch,claim));self.refs.append(kernel)
        return dict(stage=stage,parameters=names,**kernel_metadata(kernel,grid))
    def begin(self):self.check(self.lib.lt_begin(self.context));self.active=True
    def update(self,ids,budget):self.check(self.lib.lt_update(self.context,integers(ids),len(ids),budget))
    def finalize(self,ids):self.check(self.lib.lt_finalize(self.context,integers(ids),len(ids)))
    def enqueue(self,x,y,stream):self.check(self.lib.lt_enqueue(self.context,x.data_ptr(),y.data_ptr(),stream))
    def commit(self):self.check(self.lib.lt_commit(self.context));self.active=False
    def abort(self):self.check(self.lib.lt_abort(self.context));self.active=False
    def drain(self):self.check(self.lib.lt_drain(self.context))
    def stats(self):return dict(requests=int(self.lib.lt_requests(self.context)),generation=int(self.lib.lt_generation(self.context)))
    def close(self):
        if self.context:
            self.check(self.lib.lt_close(self.context));self.context=None;self.refs.clear()
