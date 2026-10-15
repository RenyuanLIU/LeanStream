"""Thin resident-buffer C ABI, no PyTorch or Python top-k on the online path."""
import ctypes as C
import numpy as np

PTR=C.c_void_p

def aligned(array):
    array=np.ascontiguousarray(array)
    backing=np.empty(array.nbytes+64,np.uint8)
    offset=(-backing.ctypes.data)%64
    result=backing[offset:offset+array.nbytes].view(array.dtype).reshape(array.shape)
    result[...] = array
    return result


def ptr(array):
    return PTR(array.ctypes.data) if array is not None else None


class Kernel:
    def __init__(self,library,precision,threads,rows=7168,stride=1664,k=215):
        self.lib=C.CDLL(str(library))
        self.lib.ls_create.argtypes=[C.c_int]*5;self.lib.ls_create.restype=PTR
        self.lib.ls_destroy.argtypes=[PTR]
        self.lib.ls_predict.argtypes=[PTR,PTR,PTR,PTR,C.c_float,PTR,C.c_int,C.c_int,PTR,PTR,PTR,PTR]
        self.lib.ls_predict.restype=C.c_int
        self.context=self.lib.ls_create(rows,stride,k,threads,precision)
        if not self.context:raise RuntimeError('Native context creation/affinity/precision failure')
        self.rows,self.stride,self.k=rows,stride,k
        self.output=aligned(np.empty(k,np.int32))
        self.debug=[aligned(np.empty(rows,np.float32)) for _ in range(3)]

    def predict(self,weights,scales,x,xscale,keep,skip=True,debug=False):
        gate,up,score=self.debug if debug else (None,None,None)
        status=self.lib.ls_predict(self.context,ptr(weights),ptr(scales),ptr(x),float(xscale),
            ptr(keep),len(keep),int(skip),ptr(self.output),ptr(gate),ptr(up),ptr(score))
        if status:raise ValueError('Native predictor status %d'%status)
        return self.output

    def close(self):
        if self.context:self.lib.ls_destroy(self.context);self.context=None

    def __del__(self):
        if getattr(self,'context',None):self.close()
