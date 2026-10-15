"""Cold pack already compiled Triton handles for the native CUDA driver backend."""
import ctypes as C
import hashlib

def pack_kernel(kernel, values):
    names = kernel.src.fn.arg_names
    key = lambda x: names.index(x) if isinstance(x, str) else x
    signature = {key(k): v for k, v in kernel.src.signature.items()}
    constants = {key(k) for k in kernel.src.constants}
    parameters = [(names[i], t) for i, t in signature.items() if i not in constants]
    packed = []
    epoch = claim = -1
    for name, ty in parameters:
        value = values[name]
        if name == 'EPOCH': epoch = len(packed)
        if name == 'CLAIM_EPOCH': claim = len(packed)
        if ty.startswith('*'): packed.append(0 if value is None else value.data_ptr())
        elif ty in ('i32', 'u32', 'i64', 'u64'): packed.append(int(value))
        else: raise ValueError('Unsupported native parameter ' + name + ': ' + ty)
    if epoch < 0: raise ValueError('Epoch specialized out of kernel')
    if kernel.metadata.num_ctas != 1: raise ValueError('Cluster launch unsupported')
    return (C.c_uint64 * len(packed))(*packed), epoch, claim, [x[0] for x in parameters]

def kernel_metadata(kernel, grid):
    return dict(grid=grid, warps_per_cta=kernel.metadata.num_warps,
                cluster_ctas=kernel.metadata.num_ctas, shared_bytes=kernel.metadata.shared,
                ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest())

class NativeBackend:
    """Own bindings; retain every compiled kernel and table through GPU retirement."""
    def __init__(self, library, offline_table=None):
        self.lib = C.CDLL(str(library))
        self.lib.ls_triton_create.restype = C.c_void_p
        self.lib.ls_triton_create_table.restype = C.c_void_p
        self.lib.ls_triton_create_table.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]
        self.lib.ls_triton_destroy.argtypes = [C.c_void_p]
        self.lib.ls_triton_bind.argtypes = [C.c_void_p,C.c_uint32,C.c_uint32,C.c_void_p,
            C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p,C.c_int,C.c_int,C.c_int]
        self.lib.ls_triton_enqueue.argtypes = [C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p]
        self._table = None
        if offline_table is None:
            self.context = self.lib.ls_triton_create()
        else:
            layers, buckets = len(offline_table), len(offline_table[0])
            if not all(len(row)==buckets for row in offline_table): raise ValueError('Ragged LUT')
            self._table = (C.c_uint8*(layers*buckets))(*(v for row in offline_table for v in row))
            self.context = self.lib.ls_triton_create_table(self._table,layers,buckets)
        if not self.context: raise MemoryError('Native backend allocation failed')
        self.lib.ls_triton_seed_claim.argtypes = [C.c_void_p,C.c_uint32]
        self._kernels = []
    def bind(self, candidate, stage, kernel, values, grid):
        args, epoch, claim, names = pack_kernel(kernel, values)
        rc = self.lib.ls_triton_bind(self.context,candidate,stage,kernel.function,grid,
            kernel.metadata.num_warps,kernel.metadata.shared,args,len(args),epoch,claim)
        if rc: raise RuntimeError('Native bind failed: '+str(rc))
        self._kernels.append(kernel)
        return dict(candidate=candidate,stage=stage,parameters=names,**kernel_metadata(kernel,grid))
    def seed_claim(self, initial):
        self.lib.ls_triton_seed_claim(self.context,initial)
    def enqueue(self, layer, io_bucket, epoch, stream):
        rc = self.lib.ls_triton_enqueue(self.context,layer,io_bucket,epoch,stream)
        if rc: raise RuntimeError('Native enqueue failed: '+str(rc))
    def close(self):
        # Caller must prove GPU completion before dropping handles or buffers.
        if self.context:
            self.lib.ls_triton_destroy(self.context)
            self.context = None
