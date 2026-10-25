"""The supported binding example is Group2Bundle, shared by built-in and edited bundles."""
from leanstream.integrations.torch.compute import Group2Bundle
# Group2Bundle cold-compiles runtime_split/persistent_split/reduce_split with WIDTH,
# PB, OFFSET and WORKERS=32, then uses existing pack_kernel and native TritonBackend.
__all__=['Group2Bundle']
