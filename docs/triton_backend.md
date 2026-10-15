# Production Triton compute backend

`backends/triton/core.py` provides resident A (`runtime_split`) and signed-INT4 cache/staging loads. `persistent.py` provides the ready-pair persistent B consumer. `reduction.py` performs the validated compact reduction. `primitives.py` retains the original rounding/publication primitives and warmup reduction. Function bodies are preserved from the validated implementation; only module imports change.

Packed group data stores gate/up/down vectors with signed low-nibble-first INT4 values and FP16 scales per 128 coordinates. The production ready-pair kernel requires Group2. It writes intermediate activations to deterministic H destinations for the application's cache-retention policy. A uses four-neuron tiles; B consumes pairs of ready logical-group descriptors, writes per-group partials, and zeros inactive partial slots. Compact reduction folds optional partial pairs without changing reduction arithmetic.

## Geometry and fallback

`include/leanstream/triton_backend.h` defines candidates with 4, 8, 16, and 32 persistent B CTAs. Candidate IDs are 0, 1, 2, and 3 respectively. Candidate 3 / 32 CTAs is the production fallback. The `workers` field is a legacy name for persistent CTA count; it is not `num_warps`, block threads, or group count. Each production CTA uses four warps / 128 threads. Triton metadata `num_ctas=1` denotes the cluster configuration, not the total grid.

For the included Ministral geometry, A=610 CTAs, B=32 persistent CTAs, reduction=128 CTAs. A grid follows resident neuron tiling; reduction grid follows width/32. Those values must not be hardcoded for another model. CUDA reference defaults are independent: its four-worker fallback has 256 threads per CTA and different per-CTA work.

## Native pre-enqueue

`backends/triton/enqueue.cpp` owns cold-bound `CUfunction` handles, fixed argument storage, epoch substitutions, and candidate-specific B specializations. Bind A and reduction once, and bind every supported B candidate before inference. The kernel specialization's compiled WORKERS constant must equal its launch grid. Changing a grid alone is invalid. Missing candidates fall back to the bound 32-CTA specialization.

The hot path chooses a candidate and submits A -> B -> optional final-prediction arm -> optional timing-disabled B-completion event -> reduction on the caller CUDA stream. This preserves the validated production submission point and maintenance ordering. It allocates no argument storage, compiles nothing, profiles nothing, and performs no synchronization or candidate search. Prediction/I/O callbacks never launch B.

`TritonBackend::view()` supplies the existing public `ComputeBackend` contract. Bound input, output, descriptor, queue, scale, activation, and partial buffers must remain valid and stable through GPU completion. Its application-provided completion callback must prove reduction completion; its cache-install callback owns cache policy. The backend owns no predictor or I/O scheduler. Serialize submissions per backend instance. Before binding a previously warmed Claim buffer, call `seed_claim` with its last warmup CLAIM_EPOCH so the first native submission uses a fresh generation. Do not copy bound backend objects.

`binding.py` packs already-compiled Triton parameters at cold setup and retains handles. `NativeBackend` exposes the same lifecycle from Python; keep the table, tensors, compiled kernels, and native library alive through retirement. It accepts an explicit offline table. No table is loaded from configs automatically. In C++, construct `TritonBackend(KernelDecisionTable)` or use `triton_selector(table)`. Supply latest caller state at pre-enqueue time; do not query per-group state or synchronize merely to obtain a newer bucket.

## Build and smoke

Use a supported CUDA toolkit, vendor PyTorch, and upstream Triton 3.1.0. Python imports need the project root on PYTHONPATH. No model is required:

```sh
cmake -S . -B build-triton -DLEANSTREAM_TRITON=ON -DLEANSTREAM_PREDICTOR=OFF
cmake --build build-triton -j2
PYTHONPATH="$PWD" python3 tests/triton_smoke.py --library "$PWD/build-triton/libleanstream_triton.so"
```

Run on an integrated CUDA device supporting coherent mapped memory and host-native atomics. The smoke warms fixed specializations, then tests native pre-enqueue, delayed prefix publication, cached and staging descriptors, exactly-once execution, original numerical tolerance, and reduction for all four candidate geometries. It does not benchmark or tune them.

The optional [Ministral adapter](../examples/ministral/README.md) preserves REPAIRED_KNEE, actual activation-based cache retention, and default unsplit attention. It routes real inference through this public native backend and these public Triton kernels.
