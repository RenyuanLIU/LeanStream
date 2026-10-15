"""Optional CPU-only predictor backing. Never used for a CPU/GPU handoff."""
import ctypes as C
import mmap
import numpy as np

class PredictorArena:
    def __init__(self,weights,scales,huge=False):
        self.align=2*1024*1024
        self.scale_offset=(weights.nbytes+63)//64*64
        size=self.scale_offset+scales.nbytes
        self.length=(size+self.align-1)//self.align*self.align
        self.storage=mmap.mmap(-1,self.length+self.align,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
        base=C.addressof(C.c_char.from_buffer(self.storage));offset=(-base)%self.align
        self.address=base+offset
        self.advice_errno=None
        if huge:
            libc=C.CDLL(None,use_errno=True);libc.madvise.argtypes=[C.c_void_p,C.c_size_t,C.c_int]
            rc=libc.madvise(self.address,self.length,14) # Linux MADV_HUGEPAGE; advisory, no system configuration.
            self.advice_errno=C.get_errno() if rc else 0
        self.weights=np.ndarray(weights.shape,dtype=weights.dtype,buffer=self.storage,offset=offset)
        self.scales=np.ndarray(scales.shape,dtype=scales.dtype,buffer=self.storage,offset=offset+self.scale_offset)
        self.weights[:]=weights;self.scales[:]=scales
        assert np.array_equal(self.weights,weights) and np.array_equal(self.scales,scales)
    def backing(self):
        # Initialization/diagnostic only; report actual THP backing, never assume advice succeeds.
        import re
        found=False;huge_kb=0
        with open('/proc/self/smaps') as f:
            for line in f:
                match=re.match(r'([0-9a-f]+)-([0-9a-f]+) ',line)
                if match:found=int(match[1],16)<=self.address<int(match[2],16)
                elif found and line.startswith('AnonHugePages:'):huge_kb=int(line.split()[1]);break
        return dict(address_alignment=self.address%self.align,length=self.length,madvise_errno=self.advice_errno,anon_huge_kb=huge_kb)
