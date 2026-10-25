# Reproduce Ministral-3 8B (asset collection v2)

This optional application consumes the model-independent LeanStream I/O, GPU and
predictor components. Generic CMake/CPU builds need no model, dataset or downloads.

## Hardware and software

Validated: NVIDIA Jetson AGX Orin, SM87; CUDA 11.8, vendor PyTorch 2.1.0a0,
Triton 3.1.0, Transformers 4.41.1, NumPy 1.24.4, safetensors 0.4.2. Also install
`tokenizers`, `huggingface_hub`, `sentencepiece`, CMake, a C++17 compiler, OpenMP,
CUDA headers/driver and pthreads. Use the vendor's matching Jetson PyTorch build;
do not replace system CUDA/PyTorch automatically. Pillow and DejaVu Sans Mono are
optional GIF dependencies. See [dependencies](dependencies.md).

Put assets/build/results on a local NVMe filesystem supporting O_DIRECT512-byte
alignment; allow at least60GB free for checkpoint, compressed archives, extracted
weights, predictor bundles, caches and results. Host RAM must accommodate mapped
staging and the loaded model. Strict bounded staging and strict physical85% cache
enforcement are not implemented. Keep power mode, CPU/GPU/EMC clocks and NVMe
placement consistent across comparisons. Scripts record observable settings and
do not change clocks or nvpmodel. Mapped coherent publication is validated on Orin;
arbitrary discrete GPUs are not claimed supported.

## Download, verify, build, run

Clone the public repository and use the matching `ministral3-8b-assets-v2` source
and Release tag. All paths below are user-selected, outside the source tree.
No predictor training, calibration, weight conversion, reordering or packing is required.

```sh
git clone https://github.com/RenyuanLIU/LeanStream.git
cd LeanStream
git checkout ministral3-8b-assets-v2
python3 -m examples.ministral.reproduce.download_model --output /data/ministral-model
python3 -m examples.ministral.reproduce.download_assets \
  --output /data/leanstream-assets --model /data/ministral-model
python3 -m examples.ministral.reproduce.prepare_coqa --assets /data/leanstream-assets --download
python3 -m examples.ministral.reproduce.verify_assets \
  --assets /data/leanstream-assets --model /data/ministral-model
python3 scripts/build_ministral.py --output /data/leanstream-build --arch sm_87
python3 -m examples.ministral.reproduce.build_baselines --output /data/leanstream-build
python3 -m examples.ministral.reproduce.smoke \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --numerical --output /data/results/numerical
python3 -m examples.ministral.reproduce.smoke \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --output /data/results/decode
```

The upstream checkpoint is independently pinned to
`cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit` revision
`e3a799844ba08f27fddbf118e154057eb0105889`. The helper downloads exact supported
configuration/tokenizer/checkpoint files and checks hashes; it does not convert weights.
The full checkpoint is not attached to the LeanStream Release.

One anonymous asset command installs nine archives: three baseline-weight parts,
two Group2-weight parts, full predictors, P1-lite predictors, metadata and the
new One-shot predictor namespace. Seven large original archives retain exact bytes;
only metadata notices were updated, and a predictor archive was added. The source
catalog, Release manifest, extracted MANIFEST.json and SHA256SUMS bind all files.
The downloader resumes partial downloads, avoids verified re-downloads, rejects
incompatible existing files and unsafe extraction, and supports `--verify-only`
and `--local-archives` for offline mirrors. No GitHub credentials are required for
public downloads. A changed/incomplete file is an error, never a silent substitute.

## Methods

**Block** retains the original1638-feature predictor on the actual normalized
post-attention MLP input. It selects2867 individual neurons, checks the immutable
2437-neuron fixed-hot cache, loads misses from SSD and waits before MLP compute.

**One-shot (v2 default)** uses1638 features padded1664, its own calibrated W4A8
parameters under `oneshot_predictors/`, and an early normalization proxy:
post_attention_layernorm applied to the current raw pre-attention residual before
Attention. It publishes one prediction and starts SSD reads early, overlapping
Attention. K2867 and immutable fixed-hot2437 are unchanged. It performs no
post-attention correction. It never loads Block's predictor as a fallback. Missing
new parameters cause a clear error. The old512-feature/raw-input implementation
is retained as historical source in fixed_cache.py but is not the default CLI.

**LeanStream** retains the validated Group2 format, active1434/keep1219 groups,
REPAIRED_KNEE, P0 budget32, P1-lite512/budget128, full dimension1664/spec-P11024,
O_DIRECT QD24, unsplit Direct-GQA, Progressive native pre-enqueue and A610/B32/
reduction128 CTAs with four warps per CTA. No example LUT, runtime profiling or
autotuning. Production Triton source is included in `backends/triton/`; users do
not download a separate kernel implementation. Native CUDA enqueue is compiled
from the public source. Block/One-shot reuse their original sparse arithmetic.

## Matched throughput

```sh
python3 -m examples.ministral.reproduce.reproduce_speed \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --method all --output /data/results/speed
```

Use `--method block`, `--method one-shot` or `--method leanstream` independently
with new output directories. `all` runs sequential independent processes. Both
forward/reverse prompt orders use exact historical sample indices0,2,5,6,7,8,9,11,
one warmup and up to16 actual greedy decode calls per prompt, stopping on EOS.
Prefill, initialization, compilation, cold binding, reset and final maintenance
drain are excluded; actual forwards through logits.argmax().item() include
Attention, CPU prediction, SSD waits and sparse compute. Pool total decode wall
milliseconds divided by actual EOS-dependent decode-call count. Do not average
rates or substitute GIF timings. comparison.json/CSV retain per-direction counts
and pooled results, per-token timing, generated IDs, feedback and KV checks.
Baseline native counters report actual miss requests/bytes and immutable hot sets.
Process-wide read_bytes/syscr are additionally sampled outside timing across
decode plus final drain; syscr includes small host/accounting reads and is not an
exact SSD request count. Predictor overhead is reported only where exposed.

Historical pooled v1 results were Block2.831, original512-feature One-shot3.281,
LeanStream8.877 tokens/s. These are historical, not v2-default measurements.
New v2 matched measurements on Jetson AGX Orin (CPU min/max2201600kHz, MAXN):

| Method | Decode calls | ms/token | tokens/s |
|---|---:|---:|---:|
| Block |230|355.527|2.813|
| One-shot (recalibrated default) |256|321.620|3.109|
| LeanStream |253|111.371|8.979|

These pool the actual forward/reverse decode wall times and EOS-dependent counts.
[Machine-readable results](../examples/ministral/reproduce/validated_results_v2.json)
retain both directions, source/native identities and storage accounting. Baseline
native SSD bytes/requests were94506774528/15381962 for Block and
103626620928/16866312 for One-shot. LeanStream process read_bytes was21813055488
and process read syscalls1775183 (includes small accounting reads; not a native
SSD-request counter). Predictor time is included in E2E; separate native predictor
timing is not exposed in this build. No historical value or GIF rate was substituted.

For baseline numerical tests use the smoke command with `--method block` or
`--method one-shot`; original rtol=.02/atol=.002 are unchanged. These tests validate
arithmetic for selected neurons, not universal selection or language quality.

## CoQA fixed500

```sh
python3 -m examples.ministral.reproduce.reproduce_accuracy \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --dry-run 5 --output /data/results/accuracy-dry5
python3 -m examples.ministral.reproduce.reproduce_accuracy \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --output /data/results/accuracy
```

This evaluates LeanStream only, using the final questions of the original500 conversations (not the official all-turn leaderboard protocol), with the original fixed500 IDs/order/prompt bytes,
greedy seed42, max64, EOS2, original per-prompt reset/warmup/cold binding, exact
historical CoQA scorer and multi-reference normalization. prepare_coqa downloads
the original licensed public dataset and reconstructs all four benchmark files to
pinned hashes; it does not select a replacement subset. Dataset contents are NOT
in Git or the asset archives. The generation job resumes only validated ordered
records with matching source/native/asset identity; an incomplete trailing record
is discarded safely. It writes generated_tokens.jsonl, decoded_answers.jsonl,
per_sample_accuracy.jsonl, accuracy_summary.json and run_manifest.json. Token
completion alone is not completed scoring. Historical Progressive F178.4321% /
EM62.3167% and Original F179.0064% /EM62.8% are historical, not same-session controls.
Previous public v1 F178.6953% /EM62.4167% is separately historical.

The recalibrated One-shot was fitted using positions from fixed500 input records0
and1. These samples were NOT held out from that evaluation set. The original predictor capture metadata also records historical calibration inputs; this reproduces that protocol, not a wholly held-out calibration dataset. No One-shot
fixed500 score is claimed here. The predictor improves some prompts but still has
serious free-generation failures. Dense/sparse training distribution shift and
missing early Attention context remain limitations; do not modify scoring rules
or filter outputs. Timing-dependent speculative reads can alter trajectories.

## Genuine generation GIF

The README animation is [actual autoregressive generation](media/autoregressive/README.md),
not a fixed-token continuation. The default One-shot now uses the exact parameter
binaries and early-input semantics recorded in that capture. The guide provides
the prompt, IDs, timestamps, EOS, identities and capture/render commands. Saved
traces render identically; a new real inference run need not produce identical
text. Older fixed-token tools remain explicitly historical diagnostics.

## License, reproducibility and troubleshooting

Project source is MIT. Model-derived weights/predictors retain Apache-2.0 and
upstream/model modification notices. MSR-LA same-terms and dated notices are also retained conservatively for calibration-derived project contributions. Project owners authorize the distributed
project contributions; this does not grant third-party dataset rights. Read the
asset LICENSE_OR_NOTICE and original dataset terms. The CoQA scorer's third-party
MIT attribution is retained in reproduce/scoring/LICENSE and THIRD_PARTY_NOTICES.md.

No private repository or unpublished binary is needed. Verify public acceptance:

```sh
python3 -m examples.ministral.reproduce.check_publication \
  --repository RenyuanLIU/LeanStream --release-tag ministral3-8b-assets-v2 \
  --output /data/results/publication-check-v2
```

This anonymously fetches the tagged source, published Release, exact checkpoint,
archives and dataset, checks hashes/layers/mapping/predictors, reconstructs CoQA
and tests documented CLI arguments. It does not run inference; follow with the
fresh Jetson build/smoke above. Keep logs outside source. On checksum errors inspect
the file rather than regenerating assets. On O_DIRECT errors use supported local
NVMe/alignment. On missing sm87 kernels use the matching compiler/environment.
Never copy private prebuilt libraries or change numerical tolerances to pass.

## Public v2 accuracy validation

The same-session LeanStream run completed all 500 records: F1 **78.4949%**, EM **61.7833%**. Relative to historical Progressive, the differences are +0.0628 and −0.5333 percentage points. All records aligned, none were missing, logits were finite, and the original scorer was used. The EM difference slightly exceeds the preferred 0.5-point comparison threshold; this timing-dependent run is reported without changing scoring or runtime behavior. These are LeanStream results, not a full-500 evaluation of the new One-shot predictor. See [`validated_results_v2.json`](../examples/ministral/reproduce/validated_results_v2.json) for measured performance, storage counters and validation provenance.
