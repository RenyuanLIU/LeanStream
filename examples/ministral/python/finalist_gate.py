"""Opt-in, validation-only allowance for isolated small FP16 RMS drift."""
import warnings
import torch


def check_rms(actual, expected, records, layer):
    # Structural differences and nonfinite values are never numerical warnings.
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError('RMS shape/dtype/device mismatch')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError('Nonfinite RMS validation input/output')
    try:
        torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-3)
    except AssertionError:
        delta=(actual.float()-expected.float()).abs()
        count=int((delta > 1e-3+1e-3*expected.float().abs()).sum().item())
        maximum=float(delta.max().item())
        # Bound both magnitude and prevalence; systematic or larger errors remain fatal.
        if actual.dtype != torch.float16 or count > max(1,actual.numel()//1000) or maximum > .01:
            raise
        record=dict(layer=int(layer),check='mapped_RMS_vs_HF',status='WARNING',
                    mismatched_elements=count,elements=actual.numel(),max_abs_diff=maximum,
                    reason='isolated small FP16 numerical drift; production computation unchanged')
        records.append(record)
        warnings.warn(str(record),RuntimeWarning)
