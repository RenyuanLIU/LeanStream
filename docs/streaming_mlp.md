# CUDA reference streaming sparse MLP

This is the standalone CUDA reference backend; production throughput evidence uses [the Triton backend](triton_backend.md).

`kernels/streaming_mlp.cu` retains the reference signed-INT4 sparse MLP arithmetic, half-rounding steps, core execution, ready-driven persistent consumer, and deterministic reduction. The release adds a launch-configuration wrapper without duplicating or modifying the mathematical implementation.

Each logical group contains `group_size` neurons. Each neuron stores gate, up, and down rows, each `width` signed INT4 values packed low/high nibble for adjacent coordinates. Group bytes are `group_size * 3 * width / 2`. FP16 scale rows use quantization groups of 128 coordinates. `width` must be a positive multiple of 256 and group size must be positive. `physical_group` locates scales; source slot locates packed bytes.

`MLPBuffers` supplies input, cache bytes, staging bytes, scales, and per-destination float partial sums. Partial storage covers all core and final consumer destinations, each of width floats. Optional execution counters and first-compute timestamps support correctness diagnostics; they are not used for candidate selection.

`enqueue_streaming_mlp` submits core -> consumer -> reduction on the caller stream. Core descriptors must be GPU-addressable before submission. Consumer metadata can be published later. Membership must preserve unique contiguous destination slots with core destinations first. The caller retains all weight and metadata storage through completion.

`enqueue_selected_streaming_mlp` performs a constant-time selector lookup at submission and forwards the chosen worker count. The v0.1 candidates use 1, 2, 4, 8, or 16 CTAs with 256 threads per CTA in the same consumer kernel. The original four-worker configuration remains the fallback. Unsupported future configuration fields also fall back to four workers in this backend.

`persistent_consumer_example` reuses the synthetic correctness suite: cached/staging descriptors, mapped readiness, shuffled and delayed publication, zero/small work sets, negative weights, exact execution counts, reduction, errors, and all five worker counts. `end_to_end_example` adds real file reads and a Runtime compute backend. Neither example benchmarks candidates.
