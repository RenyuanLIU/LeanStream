import json
from pathlib import Path
import numpy as np
from pack.int4_format import quantize, dequantize


def pack_layer(path, gate, up, down, bundle_size=4, predictor_dims=128, physical_to_logical=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    gate, up, down = [np.asarray(w, dtype=np.float16) for w in (gate, up, down)]
    i, h = gate.shape
    if up.shape != (i, h) or down.shape != (h, i):
        raise ValueError('Invalid logical MLP shapes')
    if bundle_size < 1 or not 1 <= predictor_dims <= h:
        raise ValueError('Invalid bundle size or predictor dimensions')
    order = np.arange(i) if physical_to_logical is None else np.asarray(physical_to_logical)
    if not np.array_equal(np.sort(order), np.arange(i)):
        raise ValueError('Mapping must be a permutation')
    inverse = np.argsort(order)
    features = np.linspace(0, h - 1, predictor_dims, dtype=np.int64)
    np.savez(path / 'predictor.npz', features=features, gate=gate[:, features], up=up[:, features])
    padded = ((i + bundle_size - 1) // bundle_size) * bundle_size
    scales = np.zeros((padded, 3, h // 128), dtype='<f2')
    total_error = 0.0
    maximum = 0.0
    count = 0
    with (path / 'qweights.bin').open('xb') as out:
        for start in range(0, padded, bundle_size):
            ids = order[start:start + bundle_size]
            rows = np.zeros((bundle_size, 3, h), dtype=np.float16)
            rows[:len(ids)] = np.stack((gate[ids], up[ids], down[:, ids].T), axis=1)
            packed, scale = quantize(rows)
            out.write(packed.tobytes())
            scales[start:start + bundle_size] = scale
            error = np.abs(dequantize(packed, scale)[:len(ids)].astype(np.float32) - rows[:len(ids)])
            total_error += error.sum(dtype=np.float64)
            maximum = max(maximum, float(error.max(initial=0)))
            count += error.size
    scales.tofile(path / 'scales.bin')
    meta = dict(version=0, H=h, I=i, bundle_size=bundle_size, quant_group_size=128,
                qvalue_convention='signed_twos_complement_low_nibble_first_-7_to_7',
                vector_order=['gate', 'up', 'down'], bytes_per_neuron=3*h//2,
                bytes_per_bundle=bundle_size*3*h//2, mapping_version=1,
                logical_to_physical=inverse.tolist(), padded_neurons=padded,
                predictor_features=features.tolist(), mean_absolute_error=total_error/count,
                max_absolute_error=maximum, scale_dtype='little_endian_float16')
    (path / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta
