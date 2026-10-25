"""Historical single pre-attention prediction and immutable fixed-hot cache."""
from .fixed_cache import Runtime

def install(model,assets,native,sanity=False):
    return Runtime(model,assets,native,'one_shot',sanity=sanity)
