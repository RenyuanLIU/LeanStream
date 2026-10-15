import ctypes as C
import numpy as np
from python.predictor_cpu_native import aligned, ptr

class Bridge:
    def __init__(self,library,predictor):
        self.lib=C.CDLL(str(library));p=C.c_void_p
        self.lib.si_error.restype=C.c_char_p
        self.lib.si_create.argtypes=[C.c_char_p];self.lib.si_create.restype=p
        self.lib.si_destroy.argtypes=[p]
        for name in ['si_host','si_device','si_quant']:
            f=getattr(self.lib,name);f.argtypes=[p];f.restype=p
        self.lib.si_scale.argtypes=[p];self.lib.si_scale.restype=C.c_float
        self.lib.si_epoch.argtypes=[p];self.lib.si_epoch.restype=C.c_uint32
        self.lib.si_run.argtypes=[p,C.c_int,p,p,C.c_float,p,p,p,p,p,p,p,p];self.lib.si_run.restype=C.c_int
        self.context=self.lib.si_create(str(predictor).encode())
        if not self.context:raise RuntimeError(self.lib.si_error().decode())
        self.host=np.ctypeslib.as_array((C.c_uint16*4096).from_address(self.lib.si_host(self.context))).view(np.float16)
        self.quant=np.ctypeslib.as_array((C.c_int8*1664).from_address(self.lib.si_quant(self.context)))
        self.output=aligned(np.empty(215,np.int32));self.scores=aligned(np.empty(7168,np.float32));self.times=np.empty(5,np.float64)
    def call(self,mode,item,stream,debug=False):
        device=item['gpu'] if mode=='a1' else item['residual_gpu']
        status=self.lib.si_run(self.context,0 if mode=='a1' else 1,C.c_void_p(device.data_ptr()),
           C.c_void_p(item['norm'].weight.data_ptr()),item['eps'],C.c_void_p(stream),ptr(item['features']),
           ptr(item['w']),ptr(item['scales']),ptr(item['keep']),ptr(self.output),ptr(self.times),ptr(self.scores) if debug else None)
        if status:raise RuntimeError(self.lib.si_error().decode())
        return self.times.copy()
    def close(self):
        if self.context:self.lib.si_destroy(self.context);self.context=None

