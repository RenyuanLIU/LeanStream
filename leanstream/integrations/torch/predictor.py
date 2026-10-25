"""Thin adapter to the existing CPU predictor ABI; preprocessing belongs to users."""
import ctypes as C
import numpy as np

class NativeCPUPredictor:
    """One serialized caller. input_provider returns CPU features in native precision.

    No implicit quantization, device copy, feature gathering, or model assumptions.
    Construct separate instances for different stage dimensions/budgets.
    """
    def __init__(self,library,*,weights,scales,rows,stride,budget,threads=1,precision=4,input_provider):
        if rows<=0 or stride<=0 or stride%32 or not 0<budget<=rows or precision not in (4,8,16,32):raise ValueError('Predictor geometry')
        import torch
        if torch.get_num_threads()!=threads:raise ValueError('Set torch.set_num_threads(threads) before creating the reference OpenMP predictor; team sizes must agree')
        self.lib=C.CDLL(str(library));self.precision=precision;self.stride=stride;self.budget=budget;self.input_provider=input_provider
        dtype={4:np.uint8,8:np.int8,16:np.float16,32:np.float32}[precision]
        self.weights=np.ascontiguousarray(weights,dtype=dtype);expected=rows*stride*(1 if precision==4 else 2)
        self.scales=np.ascontiguousarray(scales,dtype=np.float32)
        if self.weights.size!=expected or self.scales.size!=rows*2:raise ValueError('Predictor parameters shape')
        p=C.c_void_p;i=C.c_int
        self.lib.ls_create.argtypes=[i,i,i,i,i];self.lib.ls_create.restype=p
        self.lib.ls_destroy.argtypes=[p]
        self.lib.ls_predict.argtypes=[p,p,p,p,C.c_float,p,i,i,p,p,p,p];self.lib.ls_predict.restype=i
        self.context=self.lib.ls_create(rows,stride,budget,threads,precision)
        if not self.context:raise RuntimeError('CPU predictor creation failed')
    def __call__(self,stage,layer,activation,budget):
        if not self.context or budget!=self.budget:raise ValueError('Closed predictor or mismatched stage budget')
        features,scale=self.input_provider(stage,layer,activation)
        if hasattr(features,'device') and features.device.type!='cpu':raise ValueError('Provide asynchronously prepared CPU features; CUDA-to-CPU copy is not implicit')
        dtype={4:np.int8,8:np.int8,16:np.float16,32:np.float32}[self.precision]
        features=np.ascontiguousarray(features,dtype=dtype)
        if features.size!=self.stride or not np.isfinite(scale) or scale<=0:raise ValueError('Feature representation/scale')
        output=np.empty(budget,dtype=np.int32)
        rc=self.lib.ls_predict(self.context,self.weights.ctypes.data,self.scales.ctypes.data,features.ctypes.data,float(scale),None,0,0,output.ctypes.data,None,None,None)
        if rc:raise RuntimeError('CPU prediction failed: '+str(rc))
        return output.tolist()
    def close(self):
        if self.context:self.lib.ls_destroy(self.context);self.context=None
