# Optional Ministral-3 8B production integration

This example contains the model-specific loader, unsplit attention, P0/P1/P2 orchestration, REPAIRED_KNEE policy, and original cache-retention glue. All runtime and GPU implementation source comes from this repository. The native adapter uses `Runtime` / `WeightManager` and `backends/triton/enqueue.cpp`; A/B/compact reduction import the public `backends.triton` kernels. No private repository, binary, benchmark log, or source history is required.

The generic toolkit is independent of this model policy. This optional recipe supports the validated Group2 REPAIRED_KNEE configuration, default input/output attention, and native Progressive enqueue. Other historical model-policy experiments are not supported here. It does not expose attention checkpoints as a default inference feature.

## Dependencies

See [dependencies](../../docs/dependencies.md). The tested Jetson environment is CUDA 11.8, vendor PyTorch 2.1.0a0, upstream Triton 3.1.0, Transformers 4.41.1, NumPy 1.24.4, and safetensors 0.4.2. Use a supported environment; the build does not install or alter system packages.

## External input artifacts

Provide artifacts for the same checkpoint and calibration. They are data inputs, not hidden runtime source dependencies:

- Model directory: the supported Ministral-3 8B compressed-tensors INT4 checkpoint (`config.json`, safetensors index/shards, tokenizer files). Respect its separate model license. The validated checkpoint identity is `cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit`, revision `e3a799844ba08f27fddbf118e154057eb0105889`. Other quantized layouts are not interchangeable.
- YaRN metadata JSON: `rope_parameters` matching the checkpoint text configuration, `inv_freq` containing the authoritative inverse frequencies, and `factor` containing its attention scaling factor. Obtain these from the matching upstream model configuration/implementation; do not approximate its attention math.
- Full predictor root: `layer_00/d1664/` through `layer_33/d1664/`, each containing calibrated `runtime.npz`, `w4.bin`, `scales4.bin`, `manifest.json`, `speculative_hot.npy`, and `speculative_hot.json`. The NPZ contains `groups` (7168 x 2) and selected `features`; the native stride is 1664. The manifests bind source/checkpoint identity, hashes, and layer. W4 weights and FP32 scales follow the public predictor contract. Learned/calibrated artifacts are intentionally not bundled or retrained by this recipe.
- Compact P1 root: `layer_XX/d512/{runtime.npz,w4.bin,scales4.bin,manifest.json}` matching the full predictor identity. It supplies the original 512-feature / 128-group speculative stage. P0 budget is 32; full P2 scoring is unchanged.
- Packed weights root: `g2_weights/layer_XX/{qweights.bin,scales.bin,metadata.json}` and `g2_weights/manifest.json`. Logical/physical ordering must match each predictor's `groups`. The reader validates shape, permutation, format, and artifact identities.
- Prompts: JSONL records with `sample_id` and `input_ids`. Supply token IDs from the matching tokenizer and prompt formatting. One record suffices with `--indices 0`. No dataset is bundled or downloaded.

All required runtime source is included. Creating new learned predictor/calibration data is outside this inference release; compatible existing artifacts or a replacement predictor are required. The example fails on incompatible geometry or identities rather than silently substituting a model or predictor.

## Preparing packed weights

Use already validated artifacts when reproducing a run. For new assets, the offline helpers reuse the included checkpoint dequantizer and the original INT4 packer:

```sh
python3 examples/ministral/export_mlp.py --model /path/to/model --rope /path/to/rope.json --output /path/to/canonical
python3 examples/ministral/prepare_weights.py --canonical /path/to/canonical --predictors /path/to/full-predictors --output /path/to/weights/g2_weights
```

Canonical `layer_XX.npz` arrays are FP16 `gate[I,H]`, `up[I,H]`, and `down[H,I]`. Packing uses the predictor's unchanged Group2 ordering and the existing signed INT4/per-128 quantization. Export requires sufficient host memory and disk space. These commands create new output directories and do not alter the checkpoint or predictor inputs. They do not train predictors or determine group order.

## Build

Run from the project root:

```sh
python3 scripts/build_ministral.py --output /path/to/build-ministral --arch sm_87
export LEANSTREAM_MAPPED_LIBRARY=/path/to/build-ministral/libtriton_mapped.so
```

This builds the public runtime/I/O sources, production native enqueue, model-policy adapter, mapped-buffer helper, and reference CPU predictor. It does not borrow prebuilt LeanStream libraries. A different predictor may replace `libpredictor.so` if it implements the same ABI and artifact semantics.

## Short decode

```sh
python3 examples/ministral/run.py \
  --model /path/to/model --rope /path/to/rope.json \
  --weights /path/to/weights --predictor-root /path/to/full-predictors \
  --p1-lite-root /path/to/compact-predictors \
  --predictor-library /path/to/build-ministral/libpredictor.so \
  --shared /path/to/build-ministral/libmapped.so \
  --prompts /path/to/prompts.jsonl --indices 0 \
  --warmup-prompts 1 --max-decode-tokens 16 --no-layer-events \
  --output /path/to/new-decode-output
```

The runner selects Progressive, QD24, REPAIRED_KNEE, ordinary unsplit DirectGQA, 32 persistent B CTAs, and four warps per CTA. It warms one prompt and cold-binds compiled kernels before measured decode, preserving the validated pre-enqueue boundary. A grid is 610 and reduction grid is 128 for this model. Output includes native launch metadata. Optional `--accuracy-records` compares a few saved prefixes; it is not required for inference, builds, or normal smoke checks.

The timer brackets each ordinary model decode through greedy logits `.item()`. Prefill, reset, and final prompt drain are outside per-token timing. The existing EOS rule is preserved. No extra per-token synchronization, kernel profiling, or tuning is enabled. Use identical prompt order, clocks, power mode, predictors, and input assets for matched performance checks. The example LUT is never used for production timing.

## Short numerical check

Run `numerical.py` with the same model/weight/predictor/shared/rope/prompt/P1 arguments (omit output/indices/timing options). It performs one warmup and two decode calls with the existing all-layer numerical and descriptor checks, then verifies native geometry and exactly-once consumer counts. Correctness-only waits remain outside any performance run. No tolerances are relaxed.

This release update does not run fixed-500 accuracy. Historical Progressive F1 78.4321% / EM 62.3167% are historical context, not newly measured results. Timing-dependent I/O admission can change generated outputs even when kernel and predictor mathematics are unchanged.
