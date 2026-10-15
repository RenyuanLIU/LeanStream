# LeanStream v0.1 release checklist

## Components included

- Dynamic Weight / I/O Manager: Runtime/LayerStep, WeightManager, ExecutionQueue, STREAM_IO.
- GPU streaming computation: validated production Triton A/persistent B/compact reduction and native CUDA enqueue; separate standalone CUDA reference backend.
- Replaceable CPU predictor with public C ABI and subset support.

Optional model-policy integration lives in `examples/ministral`. Prediction schedules remain experimental and are not a fourth core component. The default production attention path is unsplit.

## Public APIs

`include/leanstream` exposes runtime/storage/queue types, predictor ABI, KernelSelector, production Triton candidate geometry, and native `TritonBackend`. Both GPU backends share the `ComputeBackend` contract. Ordered logical IDs decouple predictor policy from I/O; completion and cache-install callbacks decouple model retention from compute.

## Build commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
cmake -S . -B build-io -DLEANSTREAM_PREDICTOR=OFF
cmake --build build-io -j2
cmake -S . -B build-triton -DLEANSTREAM_TRITON=ON -DLEANSTREAM_PREDICTOR=OFF
cmake --build build-triton -j2
cmake -S . -B build-cuda -DLEANSTREAM_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build-cuda -j2
python3 scripts/build_ministral.py --output /path/to/build-ministral --arch sm_87
```

The optional model recipe and external dependency setup are documented in `examples/ministral/README.md` and `docs/dependencies.md`. No model, private source tree, old binary, or benchmark output is needed for normal component builds.

## Examples

Dynamic I/O refinement, synthetic CPU prediction, explicit offline-LUT interface, CUDA reference persistent/ready-driven execution, synthetic CUDA integration, production Triton native-enqueue numerical smoke, and optional short Ministral inference are included. The model example requires user-provided matching weights, calibrated predictor data, YaRN metadata, and tokenized prompts.

## Tests run

- Local CPU: 7/7 tests pass, including runtime, selector, prediction schedule, predictor, dynamic I/O, production import completeness, and selector contract.
- I/O-only build with predictor disabled: build and 6/6 applicable tests pass.
- CUDA reference and native Triton library: compile on x86 and Jetson for SM87.
- Jetson AGX Orin CUDA execution smoke test: PASS. Jetson CTest: 9/9 tests pass (seven CPU tests and CUDA persistent MLP / integrated runtime tests). The separate CUDA LUT integration executable also passes.
- Production Triton smoke: native pre-enqueue, delayed ready-prefix publication, cache/staging paths, exactly-once descriptors, and original numerical tolerance pass for 4/8/16/32 persistent CTAs.
- Short real-model check: all 34 layers pass the original numerical tolerances and descriptor checks; native launch metadata confirms A=610, B=32, reduction=128, four warps per CTA.

Numerical tolerances are unchanged. No Nsight, online profiling, candidate benchmark sweep, full accuracy run, or strict-memory optimization is part of this release update.

Production A/B/compact-reduction and primitive function bodies match the validated implementation. The new native wrapper preserves A -> B -> final-prediction arm -> timing-disabled completion event -> reduction and seeds the warmup claim epoch before native submission.

## Short matched production check

PASS on Jetson AGX Orin with the same model/predictors, unsplit attention, REPAIRED_KNEE, QD24, unchanged MAXN/clocks, and native pre-enqueue. Two independent forward/reverse runs per implementation use one warmup and the existing bounded eight-prompt / at-most-16-decode protocol and EOS rule.

| Implementation | Forward ms/token | Reverse ms/token | Pooled ms/token | Pooled tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Validated reference | 109.320 | 113.962 | 111.632 | 8.958 |
| Public runtime | 108.554 | 111.872 | 110.226 | 9.072 |

Public pooled latency differs by -1.26% and throughput by +1.28%; this passes the no-more-than-3%-latency-regression check. This small comparison does not establish a general speedup. The preserved EOS rule yields 255 reference and 254 public measured calls. No measured production LUT is available, so both use the validated 32-CTA fallback; the synthetic LUT is excluded. All native manifests confirm A=610, B=32, reduction=128 and four warps per CTA. Source hashes match the public source and validated reference copies. Logs and binaries remain outside the repository.

## Kernel candidates

Production Triton candidates are 4, 8, 16, and 32 persistent B CTAs. The validated production fallback is **32 persistent B CTAs**, with **four warps / 128 threads per CTA**. For the current Ministral geometry A=610 CTAs and reduction=128 CTAs. These are separate quantities, not universal grids for all models.

The CUDA reference catalog retains 1/2/4/8/16 workers and its four-worker / 256-thread fallback. That fallback is not the production Triton configuration. Standalone CUDA correctness results are not production throughput evidence.

## Kernel-selection semantics

Offline characterization -> offline table synthesis -> latest caller I/O state -> quantized bucket -> O(1) LUT lookup -> matching compiled candidate -> pre-enqueue. No online profiling or search occurs. No measured production decision LUT is distributed. Default production inference uses 32 CTAs. Synthetic example values demonstrate the API only and cannot silently override production defaults.

Each Triton B candidate requires a specialization compiled for that CTA count. An unavailable candidate falls back to the bound production default. The hot path never recompiles or changes only the launch grid of a differently specialized kernel. Runtime does not expose an exact cheap remaining-required-I/O snapshot; use caller state without introducing a wait.

## Predictor interface

`ls_create`, `ls_predict`, `ls_predict_subset`, and `ls_destroy` are declared in the public header. The original scoring/quantization mathematics are unchanged. Predictor contexts and affinity are independent from the I/O manager. The optional model-policy example loads the supplied calibrated artifacts and preserves P0/P1/P2 behavior.

## Current memory limitation

Fixed logical staging allocation remains in the reference STREAM_IO implementation. Strict bounded staging and strict physical 85% cache-capacity enforcement are not implemented. No total-memory percentage is claimed. No unfinished strict-memory branch is used.

## Accuracy status

**NOT RE-RUN FOR THIS RELEASE UPDATE.** Historical Progressive F1=78.4321% and EM=62.3167% are historical reference values only. Preserving mathematical source does not establish identical outputs under timing-dependent dynamic I/O.

## Security scan

PASS: private-path, credential-pattern, private-repository-link, and development-process scans have no problematic matches. Relative documentation links resolve. Selected project source contains no third-party notices requiring preservation that were removed; external dependency licenses are documented separately.

## Large-file scan

PASS: no file exceeds 10 MB. The public tree contains source and documentation only, with no model weights, datasets, benchmark dumps, compiled libraries/executables, object files, generated build directories, or caches.

## License status

License: MIT. See LICENSE and docs/dependencies.md. The exported project files contain no third-party copyright notices removed during extraction. Dependency source distributions are not vendored; their installation licenses and model/data distribution rights remain separate.

## Known limitations

- The production mapped-memory publication backend targets coherent integrated CUDA devices and is validated on Jetson AGX Orin.
- Triton version, CUDA toolchain, and vendor PyTorch compatibility matter; the validated dependency versions are documented.
- Model/predictor calibration artifacts and model/data licenses are external inputs.
- The generic CUDA backend has standalone correctness coverage; production model throughput uses Triton.
- The example LUT is untuned and is not production performance data.
- Strict physical cache and bounded-staging optimizations are outside v0.1.

Changes remain subject to owner review before staging, committing, or publication.
