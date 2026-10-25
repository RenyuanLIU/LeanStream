# Optional Ministral-3 8B reproduction

The [complete reproduction guide](../../docs/reproduce_ministral3_8b.md) is the recommended download → verify → build → run workflow. It covers the pinned upstream model, exact precomputed Group2 weights and calibrated predictors, original Block/One-shot baselines, Progressive inference, matched throughput, fixed-500 scoring and real-token replay.

**Artifact availability:** no public asset Release is available yet. Default public downloads stop with UNPUBLISHED. No private source or unpublished executable is required; model/experiment data still require their approved distribution route. CoQA contents are obtained under their original terms and reconstructed locally to exact historical hashes.

The supported checkpoint is `cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit`, pinned revision `e3a799844ba08f27fddbf118e154057eb0105889`. The validated environment is Jetson AGX Orin SM87, CUDA11.8, vendor PyTorch2.1.0a0, Triton3.1.0, Transformers4.41.1, NumPy1.24.4 and safetensors0.4.2. See [dependencies](../../docs/dependencies.md).

This application consumes the existing public Runtime/WeightManager, STREAM_IO, CPU predictor and production Triton/native enqueue. Model loading, unsplit Direct-GQA, P0/P1-lite/P2 orchestration, REPAIRED_KNEE, cache retention and kernel math remain unchanged. It is not a replacement for the general toolkit or its separate optional PyTorch integration.

After downloading/verifying the inputs in the guide:

```sh
python3 scripts/build_ministral.py --output /data/leanstream-build --arch sm_87
python3 -m examples.ministral.reproduce.build_baselines --output /data/leanstream-build
python3 -m examples.ministral.reproduce.smoke \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --numerical --output /data/results/numerical
```

Use `smoke --method block` or `--method one-shot` for the original bounded baseline checks; omit `--numerical` for short normal Progressive decode. Production B is32persistent CTAs with four warps/CTA; the synthetic example LUT is never loaded.

Existing `run.py`, `numerical.py`, `python/`, `native/` and `scripts/` remain available unchanged. Developer export/packing helpers remain in the tree for new research artifacts, but reproducing the supported experiment must use the exact validated downloadable binaries. It does not require training, calibration, quantization, reordering, packing or fixed-hot generation.

Previous measurements are evidence, not expected constants: matched public-wrapper Block2.831, One-shot3.281 and LeanStream8.877tokens/s; completed public fixed-500 F1 78.6953%, EM62.4167%. Their protocols and distinct historical references are documented in the guide; these are historical measurements, not expected outputs.
