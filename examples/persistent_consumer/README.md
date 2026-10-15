# CUDA reference persistent consumer example

Build with `-DLEANSTREAM_CUDA=ON`, then run `./build-cuda/persistent_consumer_example` on a compatible GPU. It shares the synthetic correctness implementation in `tests/persistent_mlp_test.cu` and the single mathematical kernel in `kernels/streaming_mlp.cu`.

The example uses separate synthetic cache/staging buffers, mapped ready descriptors, progressive publication, reduction, and an independent small arithmetic check. It exercises all five worker configurations and the original batch/progressive equivalence checks. Delayed publication checks readiness semantics, not performance. No checkpoint or production benchmark is required.

This CUDA reference test is separate from the validated production Triton execution path. See [production Triton](../../docs/triton_backend.md).
