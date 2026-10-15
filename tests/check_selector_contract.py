"""Keep the selection path independent of GPU timing and dynamic search."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['include/leanstream/kernel_selector.h','src/runtime/kernel_selector.cpp',
       'kernels/streaming_mlp_candidates.cu']
for name in paths:
    text=(root/name).read_text().lower()
    for forbidden in ('cudaevent', 'autotune_layer', 'autotune_all_layers',
                      'onlineprofiler', 'runtimetuner', 'autotuner', 'benchmark'):
        assert forbidden not in text, (name,forbidden)
print('PASS: selector API and enqueue helper contain no profiling or tuning calls')
