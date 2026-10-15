"""Mapped tensor publication helpers for coherent integrated CUDA devices."""
import ctypes as C
from pathlib import Path
import numpy as np

class Mapped:
    def __init__(self,lib,count,dtype):
        import torch
        self.lib=lib;self.count=count;self.dtype=np.dtype(dtype)
        self.context=lib.tm_alloc(count*self.dtype.itemsize);assert self.context,'Mapped allocation failed'
        self.host=np.ctypeslib.as_array((C.c_uint8*(count*self.dtype.itemsize)).from_address(lib.tm_host(self.context))).view(dtype)
        capsule=C.pythonapi.PyCapsule_New;capsule.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];capsule.restype=C.py_object
        code=2 if self.dtype.kind=='f' else 0
        self.tensor=torch.utils.dlpack.from_dlpack(capsule(lib.tm_dlpack(self.context,code,8*self.dtype.itemsize,count),b'dltensor',None))
    def release(self,epoch):self.lib.tm_release(self.context,0,epoch)
    def wait(self,epoch):
        if self.lib.tm_acquire_all(self.context,self.count,epoch,5000):raise RuntimeError('Mapped acquire timeout')
    def close(self):del self.tensor;self.lib.tm_free(self.context)

def library(root):
    path=Path(root)
    if not path.is_file():raise RuntimeError("Build the native Triton support library first")
    l=C.CDLL(str(path.resolve()));p=C.c_void_p;i=C.c_int
    for name,args,result in [('tm_alloc',[C.c_uint64],p),('tm_free',[p],None),('tm_host',[p],p),('tm_device',[p],p),
        ('tm_dlpack',[p,i,i,C.c_int64],p),('tm_release',[p,i,C.c_uint32],None),('tm_acquire_all',[p,i,C.c_uint32,i],i)]:
        f=getattr(l,name);f.argtypes=args;f.restype=result
    return l
