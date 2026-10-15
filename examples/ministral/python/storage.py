import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from pack.int4_format import dequantize


class LayerReader:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        m = self.meta
        if m['version'] != 0 or m['mapping_version'] != 1 or m['quant_group_size'] != 128:
            raise ValueError('Unsupported format')
        if (m['qvalue_convention'] != 'signed_twos_complement_low_nibble_first_-7_to_7'
                or m['vector_order'] != ['gate','up','down']
                or m['scale_dtype'] != 'little_endian_float16'
                or m['H'] % 128 or m['bundle_size'] < 1
                or m['bytes_per_neuron'] != 3*m['H']//2
                or m['bytes_per_bundle'] != m['bundle_size']*m['bytes_per_neuron']
                or m['padded_neurons'] != ((m['I']+m['bundle_size']-1)//m['bundle_size'])*m['bundle_size']):
            raise ValueError('Inconsistent format metadata')
        self.mapping = np.asarray(m['logical_to_physical'], dtype=np.int64)
        if not np.array_equal(np.sort(self.mapping), np.arange(m['I'])):
            raise ValueError('Invalid mapping')
        self.scales = np.fromfile(self.path / 'scales.bin', dtype='<f2').reshape(m['padded_neurons'], 3, m['H']//128)
        if not np.isfinite(self.scales).all() or np.any(self.scales <= 0):
            raise ValueError('Invalid scales')
        expected = m['padded_neurons'] * m['bytes_per_neuron']
        if (self.path / 'qweights.bin').stat().st_size != expected:
            raise ValueError('Wrong qweight file size')
        self.fd = os.open(self.path / 'qweights.bin', os.O_RDONLY)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        m = self.meta
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= m['I']) or len(np.unique(ids)) != len(ids):
            raise ValueError('Expected unique valid logical IDs')
        physical = self.mapping[ids]
        bundles = np.unique(physical // m['bundle_size'])
        start = perf_counter()
        records = {}
        for bundle in bundles:
            size = m['bytes_per_bundle']
            data = os.pread(self.fd, size, int(bundle)*size)
            if len(data) != size:
                raise IOError('Short bundle read')
            records[int(bundle)] = np.frombuffer(data, dtype=np.uint8).reshape(m['bundle_size'], 3, m['H']//2)
        read_ms = (perf_counter() - start)*1000
        start = perf_counter()
        packed = np.empty((len(ids), 3, m['H']//2), dtype=np.uint8)
        for n, p in enumerate(physical):
            packed[n] = records[int(p)//m['bundle_size']][int(p)%m['bundle_size']]
        rows = torch.from_numpy(dequantize(packed, self.scales[physical])) if len(ids) else torch.empty((0,3,m['H']), dtype=torch.float16)
        physical_bytes = len(bundles)*m['bytes_per_bundle']
        useful = len(ids)*m['bytes_per_neuron']
        return rows, dict(logical_requested=len(ids), bundles_read=len(bundles), physical_bytes=physical_bytes,
                          useful_logical_bytes=useful, read_amplification=physical_bytes/useful if useful else 0,
                          read_ms=read_ms, dequant_ms=(perf_counter()-start)*1000)
