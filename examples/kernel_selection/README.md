# CUDA reference kernel selection examples

`./build/kernel_selection_example` prints candidate configurations and a synthetic three-layer/eight-bucket LUT's decisions for zero, half, and full remaining load. It also demonstrates out-of-range fallback. It performs no benchmark.

The optional CUDA target `kernel_selection_cuda_example` shares the synthetic end-to-end integration source:

```sh
python3 tests/run_io_example.py ./build-cuda/kernel_selection_cuda_example cuda
```

Selection lives inside `ComputeBackend::enqueue`, before core/consumer/reduction submission. Its bucket zero is illustrative caller-supplied state, not a measured remaining-I/O count. See [kernel selection contract](../../docs/kernel_selection.md).

Production Triton uses a separate candidate catalog and 32-CTA fallback. The synthetic table in this example does not configure production inference. See [production Triton](../../docs/triton_backend.md).
