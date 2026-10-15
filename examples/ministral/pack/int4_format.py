"""Signed two's-complement nibbles, first value in the low nibble."""
import numpy as np


def quantize(rows, group_size=128):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.shape[-1] % group_size or group_size % 2:
        raise ValueError('Vector length must be divisible by an even group size')
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite weights')
    groups = rows.reshape(*rows.shape[:-1], -1, group_size)
    scale = (np.max(np.abs(groups), axis=-1) / 7).astype(np.float16)
    scale = np.maximum(scale, np.float16(np.nextafter(np.float16(0), np.float16(1))))
    if not np.isfinite(scale).all():
        raise ValueError('FP16 scale overflow')
    q = np.clip(np.rint(groups / scale[..., None].astype(np.float32)), -7, 7)
    q = q.astype(np.int8).reshape(rows.shape).view(np.uint8) & 15
    return q[..., ::2] | (q[..., 1::2] << 4), scale


def dequantize(packed, scales, group_size=128):
    packed = np.asarray(packed, dtype=np.uint8)
    q = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int8)
    q[..., ::2] = (packed & 15).astype(np.int8)
    q[..., 1::2] = (packed >> 4).astype(np.int8)
    q[q >= 8] -= 16
    groups = q.reshape(*q.shape[:-1], -1, group_size)
    return (groups.astype(np.float32) * np.asarray(scales)[..., None]).reshape(q.shape).astype(np.float16)
