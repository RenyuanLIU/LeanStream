# Porting and integration

Start with CPU tests and synthetic examples. Linux aligned file I/O, pthreads, and OpenMP are required by the reference transport/predictor. CPU inference components need no model checkpoint. ARM predictor builds require ARMv8.2 FP16 and dot-product instructions. CUDA is opt-in; choose a supported architecture explicitly.

Define stable logical group IDs and a `GroupLayout` mapping to physical groups, byte offsets, and aligned extents. The runtime does not depend on one model architecture. Pack weights and scales according to the included MLP's documented layout, or supply a different compute backend. Changing file ordering must not change logical membership.

Allocate staging for every logical slot in v0.1. Supply cache-slot bindings from the application's residency policy. Keep all borrowed storage alive until GPU and callbacks retire. Enforce any physical memory budget outside this reference backend; strict capacity enforcement and bounded staging are separate ongoing optimizations.

CUDA consumers require system-scope acquire instructions, coherent mapped host metadata/weights, and correct CPU-to-GPU release/acquire visibility. Do not assume arbitrary pageable host allocations are GPU-addressable. Check device support, mapping, alignment, and visibility for your platform before deployment. The supplied tests use mapped pinned buffers and device allocations. Error termination invalidates output even if a reduction kernel has run.

Implement `ComputeBackend` enqueue/complete/commit callbacks. Enqueue core, consumer, and reduction in advance on the supplied stream. Completion proves lifetime, not candidate performance. Commit installs actual cache bytes after GPU use before allowing retirement. Deferred commit must copy/retain IDs and invoke retirement exactly once after all users finish. Keep cancellation valid even after partial enqueue failures.

Place selector policy in the compute backend. Production Triton uses `TritonBackend` with its 32-CTA fallback; the CUDA reference can use `enqueue_selected_streaming_mlp`. Do not substitute its four-worker geometry for the production configuration. Feed cheap application-owned I/O state at submission time. Characterize configurations and synthesize the LUT offline. Always keep the default fallback. Do not add predictor waits, I/O waits, or GPU synchronization to obtain a newer selection state.

Replace prediction policy with any backend returning ranked logical IDs. Preserve immediate I/O refinement, non-preemptive started reads, READY reuse, final-only membership, epoch ownership, and callback retirement. Validate sparse arithmetic against an independent reference for your packed format before assessing performance.

License: MIT. See [LICENSE](../LICENSE) for terms. No explicit third-party copyright or license notices were found in the selected source files; this observation is not a complete provenance audit. Review CUDA and OpenMP toolchain/runtime distribution requirements if distributing binaries.

For model-dependent integration, [the Ministral recipe](../examples/ministral/README.md) includes all required source and describes external weight, predictor, and prompt inputs. [Dependency licenses and environment setup](dependencies.md) are separate from the project MIT license.
