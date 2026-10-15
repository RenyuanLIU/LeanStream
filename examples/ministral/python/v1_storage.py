"""Synchronous buffered pread, requested-row gathering, separate GPU dequant."""
from time import perf_counter
import os
import numpy as np
import torch
from python.storage import LayerReader


def read_runs(bundles, gap=0):
    if gap < 0:
        raise ValueError('Negative collapse gap')
    runs = []
    for b in sorted(set(map(int,bundles))):
        if runs and b-runs[-1][1]-1 <= gap:
            runs[-1][1] = b
        else:
            runs.append([b,b])
    return runs


def gpu_dequantize(packed, scales):
    """Float32 scale multiplication then FP16 rounding, identical to CPU format."""
    if packed.device.type != 'cuda' or scales.device != packed.device:
        raise ValueError('GPU dequant requires CUDA qweights and scales')
    if packed.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError('Expected uint8 qweights and FP16 scales')
    n = packed.shape[-1]*2
    if n % 128 or scales.shape != (*packed.shape[:-1],n//128):
        raise ValueError('Invalid quantization shape')
    q = torch.stack((packed & 15,packed >> 4),dim=-1).reshape(*packed.shape[:-1],n).to(torch.int8)
    q = torch.where(q >= 8,q-16,q)
    return (q.reshape(*q.shape[:-1],n//128,128).float()*scales.unsqueeze(-1).float()).reshape(q.shape).half()


class PackedReader(LayerReader):
    def __init__(self,path,collapse_gap=0,bundle_size=None):
        super().__init__(path)
        if collapse_gap < 0:
            raise ValueError('Negative collapse gap')
        self.collapse_gap = collapse_gap
        if bundle_size is not None and bundle_size != self.meta['bundle_size']:
            raise ValueError('Bundle override requires repacked artifact with matching metadata')
        self.device_scales = None

    def read_packed(self,ids):
        ids = np.asarray(ids,dtype=np.int64)
        m = self.meta
        if ids.ndim != 1 or np.any(ids<0) or np.any(ids>=m['I']) or len(np.unique(ids)) != len(ids):
            raise ValueError('Expected unique valid logical IDs')
        physical = self.mapping[ids]
        bundles = np.unique(physical//m['bundle_size'])
        runs = read_runs(bundles,self.collapse_gap)
        records = []
        start = perf_counter()
        for lo,hi in runs:
            size = (hi-lo+1)*m['bytes_per_bundle']
            data = os.pread(self.fd,size,lo*m['bytes_per_bundle'])
            if len(data) != size:
                raise IOError('Short collapsed read')
            records.append((lo,hi,data))
        io_ms = (perf_counter()-start)*1000
        start = perf_counter()
        packed = np.empty((len(ids),3,m['H']//2),dtype=np.uint8)
        # Find each requested row's run once; avoid rescanning all IDs per run.
        views = [np.frombuffer(data,dtype=np.uint8).reshape(-1,3,m['H']//2) for _,_,data in records]
        if len(ids):
            run_index = np.searchsorted([hi for _,hi,_ in records],physical//m['bundle_size'])
            for dest,(position,run) in enumerate(zip(physical,run_index)):
                packed[dest] = views[run][position-records[run][0]*m['bundle_size']]
        tensor = torch.from_numpy(packed)
        prepare_ms = (perf_counter()-start)*1000
        useful = len(ids)*m['bytes_per_neuron']
        physical_bytes = sum((hi-lo+1)*m['bytes_per_bundle'] for lo,hi in runs)
        return tensor, physical, dict(missing_logical_neurons=len(ids),useful_requested_qweight_bytes=useful,
            physical_bytes_read=physical_bytes,overread_bytes=physical_bytes-useful,pread_calls=len(runs),
            read_amplification=physical_bytes/useful if useful else 0,io_ms=io_ms,
            physical_io_MBps=physical_bytes/io_ms/1000 if io_ms else 0,
            useful_io_MBps=useful/io_ms/1000 if io_ms else 0,packed_prepare_ms=prepare_ms)

    def scales_on(self,device):
        if self.device_scales is None:
            self.device_scales = torch.from_numpy(self.scales).to(device)
        return self.device_scales
