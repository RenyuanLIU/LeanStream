# Reproduce the Ministral-3 8B results

**Artifact availability: public asset downloads are not yet available.** The default downloader reports UNPUBLISHED until the checksummed assets are published in a public GitHub Release. This guide is the recommended reproduction entry point; the generic LeanStream toolkit and PyTorch integration remain independent.

## Platform and environment

Validated hardware: NVIDIA Jetson AGX Orin, SM87, 32 GB-class coherent unified memory, 12 CPU cores, local NVMe SSD supporting O_DIRECT/512-byte aligned reads. Arbitrary discrete GPUs are not supported by the current mapped-memory publication contract. Keep assets/builds/results outside the checkout, on SSD rather than tmpfs or network storage. Allow at least 60 GB free for archives, extraction, upstream download/cache, builds and results.

| Dependency | Validated version |
| --- | --- |
| CUDA | 11.8 |
| Vendor PyTorch | 2.1.0a0+41361538.nv23.6 |
| Triton | 3.1.0 |
| Transformers | 4.41.1 |
| NumPy | 1.24.4 |
| safetensors | 0.4.2 |
| tokenizers | 0.19.1 |
| huggingface_hub | 0.23.1, checkpoint downloader only |

C++17, g++, OpenMP, pthreads, CUDA driver/runtime development files and Python3.8+ are required. Archive handling uses Python's standard gzip/tar libraries; no system zstd or GitHub CLI is required. Pillow and DejaVu Sans Mono are optional GIF dependencies. See [dependencies](dependencies.md); do not replace vendor PyTorch or install system packages automatically.

Historical matched runs used MAXN mode0, CPU min/max2,201,600kHz and GPU min/max1,300,500,000Hz. EMC was not readable without elevated access. The scripts record observable hardware/software, clocks, power mode and SSD mount; they do not change any settings. Use the same settings and SSD for all methods. Do not claim unobserved EMC clocks are pinned.

```sh
export PYTHONDONTWRITEBYTECODE=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
export HF_HOME=/data/hf-cache
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Use the Python interpreter containing the validated dependencies. All module commands below run from the repository root. Builds, artifact validation and imports never automatically run model evaluation.

## 1. Download the pinned upstream checkpoint

The exact supported model is `cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit`, revision `e3a799844ba08f27fddbf118e154057eb0105889`. Despite the repository's AWQ name, its validated loader consumes the checkpoint's compressed-tensors INT4 structure; another AWQ/Mistral checkpoint is not interchangeable.

```sh
python3 -m examples.ministral.reproduce.download_model --output /data/ministral-model
python3 -m examples.ministral.reproduce.download_model --output /data/ministral-model --verify-only
```

The helper uses immutable Hub revision and per-file SHA256/size identities from `artifact_catalog.json`. Existing valid files are skipped; incompatible checkpoint files are never overwritten. Hub partial downloads resume in its cache. This download is separate from LeanStream assets; the complete upstream checkpoint is not republished in our archives. Its expected local files total14,386,869,646bytes.

## 2. Download/extract exact LeanStream assets

```sh
# Available only after the approved public asset Release is published:
python3 -m examples.ministral.reproduce.download_assets --output /data/leanstream-assets
```

**Current status: this public command stops with UNPUBLISHED.** The intended public repository is `RenyuanLIU/LeanStream`, proposed asset tag `ministral3-8b-assets-v1`. These are planned destinations, not claims that Release attachments exist. No fake URL or fallback artifact is used.

For offline use, supply the matching archives and checked-in `asset_manifest.json`:

```sh
python3 -m examples.ministral.reproduce.download_assets \
  --local-archives /data/approved-local-packages --output /data/leanstream-assets \
  --model /data/ministral-model
python3 -m examples.ministral.reproduce.download_assets \
  --output /data/leanstream-assets --model /data/ministral-model --verify-only --scope runtime
```

`--release-tag`, `--repository`, `--release-manifest` and `--cache` select an explicit version/location. Public downloads use ordinary HTTPS and require no GitHub CLI.

Archive downloads support HTTP Range resume and final size/SHA256 verification. Verified cached archives and extracted files are reused. Servers that ignore Range cause a complete restart of that partial file, never an invalid append. Corrupt or incompatible existing complete files fail explicitly. Extraction rejects traversal, links, duplicate/unexpected members, wrong member sizes and file checksum mismatches. No predictor training, calibration, weight quantization, reordering, packing or hot-set generation occurs.

### Exact collection and provenance

`artifact_catalog.json` is the original per-file identity inventory. `asset_manifest.json` binds its SHA256 to eight independent `.tar.gz` archives, archive sizes/checksums, complete extracted paths, model geometry and compatibility. Every archive is below1.8GiB. The current compressed archive total is6,685,344,829bytes; dataset-derived benchmark files and the upstream checkpoint are excluded.

The collection contains all34layers of:

- Exact reordered/requantized Group2 signed INT4 weights, per128 FP16 scales and logical/physical mappings: `leanstream_weights/g2_weights/`.
- Original full d1664 predictors, including runtime/packed parameters, manifests and speculative hot order: `full_predictors/`.
- Matching compact d512 P1-lite predictors: `p1_lite_predictors/`.
- Original individual-neuron baseline SSD weights, predictors, scales and immutable hot IDs: `baseline_weights/`.
- Authoritative YaRN inverse frequencies, attention scaling, model/checkpoint identity, tokenizer compatibility and original protocol metadata: `metadata/`.

The original Group2 manifest SHA256 is `2a386a8a76b4fed1fb51b4b9af43a24daa083c00ecc72730b862d4252ed7d2e9`. Its logical groups follow the validated predictor permutation, not file order. Original weight binaries are copied byte-for-byte; newly repacked substitutes are rejected. YaRN includes64inverse frequencies, frequency-extension factor16 and attention scaling factor1.0, checked against the pinned checkpoint parameters; these factors are different quantities.

## 3. Reconstruct the exact historical benchmark inputs

**CoQA text, references and prompt token IDs are not redistributed in our archives.** Obtain the original public dataset under its source terms, then use the deterministic local recipe. This is the sole dataset-preparation step; it does not select a different subset or change tokenization.

```sh
# Explicit download of the original licensed dev-v1.0 dataset:
python3 -m examples.ministral.reproduce.prepare_coqa --assets /data/leanstream-assets --download
# Or use your own exact copy:
python3 -m examples.ministral.reproduce.prepare_coqa \
  --assets /data/leanstream-assets --dataset /data/coqa-dev-v1.0.json
python3 -m examples.ministral.reproduce.verify_assets \
  --assets /data/leanstream-assets --model /data/ministral-model
```

The original dataset SHA256 is `dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a`, 9,090,845bytes. The official Stanford endpoint is primary; a byte-identical EleutherAI mirror is pinned to revision `dd90bb77780a0cacf024a05ce08874930cd2b22d`. The newer stripped CoQA mirror lacks IDs/additional answers and is not substituted.

`coqa_selection.json` contains only exact sample IDs/order/source/turn counts and expected hashes. Each sample targets the final question of one conversation with its preceding history; this is not the official all-turn CoQA test protocol. The original500validation conversations include100each from MCTest, RACE, CNN, Wikipedia and Gutenberg. The recipe restores original multi-reference fields, uses unchanged `scoring.coqa.doc_to_text`, the exact tokenizer compatibility JSON with BOS, no truncation, and checks **byte identity** of all four historical benchmark files before installing them:

- prompt IDs SHA256: `9a04827b86633cb59f7ec6e4a30ed22b52e67073c5afd08179f4e0a294112093`;
- frozen documents SHA256: `f2e8cff78159f98eef5708c0b09243b35120c0463483ffd557abe4a452c8653c`.

The old experiment did not record an upstream dataset Git revision. The recovered exact raw-file checksum, pinned byte-identical mirror and complete reconstruction now provide a verifiable public input route; no historical revision is invented. All old prompt/scorer semantics remain unchanged.

## 4. Build and run short checks

Production Triton source is public: [resident arithmetic](../backends/triton/core.py), [persistent READY consumer](../backends/triton/persistent.py), [reduction](../backends/triton/reduction.py), and [native CUDA enqueue](../backends/triton/enqueue.cpp). Model-specific adapters remain in `examples/ministral/`; the standalone CUDA kernels are a separate reference backend.

```sh
python3 scripts/build_ministral.py --output /data/leanstream-build --arch sm_87
python3 -m examples.ministral.reproduce.build_baselines --output /data/leanstream-build
python3 -m examples.ministral.reproduce.smoke \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --output /data/results/numerical --numerical
python3 -m examples.ministral.reproduce.smoke \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --output /data/results/decode
```

Use new output directories. `smoke --method block` and `smoke --method one-shot` invoke the original bounded baseline numerical/descriptor checks. LeanStream numerical smoke delegates to unchanged `examples/ministral/numerical.py`; normal short decode delegates to unchanged `run.py`. No historical prebuilt library is used. Native geometry must be A610/B32/reduction128, four warps per CTA. A B grid of4 is invalid for this production configuration. Existing detailed entry points remain available for developers but are not a second artifact-preparation workflow.

## 5. Three-method throughput

```sh
python3 -m examples.ministral.reproduce.reproduce_speed \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --method all --output /data/results/speed
```

Individual methods use `--method block`, `--method one-shot`, or `--method leanstream`. Methods run sequentially, one independent process per direction/method. The order is forward Block/One-shot/LeanStream, then reverse Block/One-shot/LeanStream. Each process warms one prompt before measured calls. Prompt indices are `0,2,5,6,7,8,9,11` and their reverse; each permits up to16decode calls, stopping when the preceding token is EOS2. Prefill's first selected token is not counted as a decode call.

Per-token timing brackets the actual model decode through greedy logits `.item()`, including normal attention, CPU prediction, SSD waits, weight management, sparse compute and critical-path synchronization. Initialization, cold compilation/binding, warmup, prefill, reset and final maintenance drain are excluded. Pooled latency is total measured decode milliseconds / total decode calls; throughput is1000/pooled latency. Output includes each direction, per-token CSV, generated IDs, EOS-dependent counts, provenance and JSON/CSV comparison. No timing result is hardcoded.

### Method semantics

**Block:** current normalized post-attention MLP input, 1,638features padded1664, CPU W4A8 prediction, select2,867individual neurons, immutable2,437-neuron fixed-hot cache, required O_DIRECT reads/QD24, wait for all misses, then original sparse GPU computation. No early prefetch or current required-I/O/MLP overlap.

**One-shot:** one512-feature prediction from the raw pre-attention layer input, early SSD prefetch overlapping attention, required-I/O wait at MLP, then original sparse computation. No progressive refinement, insertion, eviction or adaptive hot-set changes. It shares Block's original hot IDs and individual-neuron 6,144-byte SSD format. These predictor inputs differ by design; this is not a pure overlap-only ablation.

Both baseline implementations are recovered from historical `e313b4d…` / `9533c348…`; [source checksums/notices](../examples/ministral/reproduce/THIRD_PARTY_NOTICES.md) identify them. They do not use LeanStream dynamic cache retention.

**LeanStream:** Group2, active1434/keep1219groups, calibrated full d1664 predictor, P0budget32, P1-lite512features/budget128, speculative P1size1024, REPAIRED_KNEE, QD24/O_DIRECT, unsplit Direct-GQA, native pre-enqueued Progressive A/B/reduction, B32CTAs and warps4. No synthetic LUT, online profiling, candidate search or autotuning.

Historical protocol distinction: the older baseline launcher forced16calls after EOS and used forward/reverse in one process. The matched public-wrapper measurements below use the common EOS-aware independent-process protocol described above. Do not relabel older numbers as same-protocol measurements.

| Method | Matched public-wrapper ms/token | Tokens/s | Decode calls |
| --- | ---: | ---: | ---: |
| Block |353.222|2.831|230 |
| One-shot |304.773|3.281|226 |
| LeanStream |112.646|8.877|256 |

These are previously completed measurements, retained in `validated_results.json`, not expected outputs for a new run. Historical approximate values were2.9/3.2/9.0tok/s. Another separate public run measured110.226ms/token /9.072tok/s. Hardware, thermal state, SSD and timing-dependent speculative reads affect results; no implementation is tuned to reproduce a target value.

## 6. Optional fixed-500 CoQA accuracy

```sh
python3 -m examples.ministral.reproduce.reproduce_accuracy \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --dry-run 5 --output /data/results/accuracy-dry
# Explicit full evaluation, not a default build/test action:
python3 -m examples.ministral.reproduce.reproduce_accuracy \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --output /data/results/accuracy
```

Use distinct dry/full output directories. The original protocol is greedy, batch1, seed42, max_new_tokens64, EOS2, first-prompt prefill/four-decode warmup, drain, cold native binding and per-prompt reset. Generation stops at EOS/64. The original compatible decoder stops the scored prefix at decoded `\nQ:`, EOS or64; the scorer retains original first-line extraction, Transformers SQuAD normalization and multi-reference leave-one-out max averaging. Scorer SHA256 is `62ac82e8027162d134709355a00526ff5497deb5987beebb18b439b320e860ca`; it is unchanged, not replaced by single-reference EM.

Resume validates source/native/artifact identities and every completed sample's index, ID, token range and stopping condition. A partial final line can be discarded only after its complete prefix validates. Existing valid samples are not duplicated. Outputs are `generated_tokens.jsonl`, decoded answers, per-sample scores, aggregate summary and run manifest. TOKENS_COMPLETE alone is not scored accuracy.

| Accuracy | F1 | EM | Provenance |
| --- | ---: | ---: | --- |
| Original |79.0064%|62.8000%|Historical |
| Progressive |78.4321%|62.3167%|Historical |
| Public500/500 |78.6953%|62.4167%|Previously completed public evaluation, reused |

The public result is +0.2632pp F1 / +0.1000pp EM versus historical Progressive. Timing-dependent I/O means token trajectories need not be identical across runs; do not change scoring or runtime policy to force agreement.

## 7. Fixed-token decode performance GIF

The README animation measures one prompt and the same 112-token reference continuation through all three original methods. Each teacher-forced token is fed into a real model forward with KV cache. The returned next-token prediction is recorded but never fed back. The displayed text is reference input, not a generated answer. All runs use identical timing boundaries and one common replay scale; historical pooled free-generation throughput remains separate.

```sh
python3 -m examples.ministral.reproduce.capture_fixed_tokens \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --input-file docs/media/fixed_decode_input.json --output /data/results/fixed-decode
python3 tools/make_speed_gif.py \
  --trace /data/results/fixed-decode/fixed_token_decode_trace.json \
  --output /data/results/fixed-decode.gif
```

The processes run sequentially with unchanged predictors, SSD reads, cache policies, sparse kernels and Progressive native bindings. See [exact inputs, measured timing and rendering](media/README.md). The optional `capture_tokens` command still supports free-generation diagnostics and `--prompt-file`; those traces remain available for quality analysis and do not drive this performance visualization.

## Licenses and publication state

The [pinned checkpoint](https://huggingface.co/cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit/tree/e3a799844ba08f27fddbf118e154057eb0105889) declares Apache-2.0. Model-derived packed/requantized/reordered weights require applicable license/attribution notices and modification identification. The project authors created the calibrated predictors and fixed-hot artifacts. Their public distribution license and approval for public redistribution of the collection remain pending; LeanStream's MIT source license does not automatically license these data files. Public publication requires the owner's explicit approval of the collection and final notices. The checkpoint itself is downloaded from upstream, not redistributed here.

[CoQA's source terms](https://stanfordnlp.github.io/coqa/#license) are mixed CC BY-SA4.0, MSR-LA, RACE and CNN/Apache terms. We do not claim blanket redistribution permission or upload CoQA data. The explicit download/reconstruction route preserves these terms and exact historical inputs. No weight/predictor preprocessing is required.

**Public asset Release: unavailable.** Independent download-and-run reproduction remains blocked until the artifact licenses/notices are approved and the complete checksummed collection is published. The downloader does not substitute other weights or predictors.

## Verify a published source and artifact Release

After the public source tag and asset Release actually exist, run the read-only acceptance check in a new output directory:

```sh
python3 -m examples.ministral.reproduce.check_publication \
  --repository RenyuanLIU/LeanStream --release-tag ministral3-8b-assets-v1 \
  --output /data/results/publication-check
```

This performs real anonymous downloads of tagged source, all eight archives/control files and the pinned upstream checkpoint. Allow at least 60 GB free and install the optional reproduction dependencies first. It does not use GitHub tokens, gh credentials or an existing private asset directory. It executes the downloader from the tagged source with credentials removed, checks Release attachment sizes and SHA256SUMS, verifies catalog identities after fresh extraction, and checks that documented CLI flags exist. The tagged source must preserve the validated inference files from the manifest's publicly accessible supported source revision. Publication metadata must already be approved and consistent; the checker never changes it.

A Draft Release, missing tag, unavailable assets, pending redistribution approval or wrong checksums produces **BLOCKED**, not anonymous-download success. Results and download logs are written to `publication_check.json` and companion logs under the requested directory. A successful check verifies public distribution and artifact identities; run the separate Jetson build/numerical/decode smoke commands above to establish GPU execution. It never launches a benchmark or fixed-500 evaluation automatically. **The public Release is currently unavailable, so this check cannot yet pass.**

## Tests and troubleshooting

```sh
USE_TF=0 PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s examples/ministral/reproduce/tests
```

Generic CTest remains model-independent. A checksum mismatch is an incompatible input, not permission to repack/retrain. Missing benchmark files require the exact `prepare_coqa` step. Wrong tokenizer/model revision fails. O_DIRECT errors require suitable aligned local SSD storage. Module/native provenance must resolve to this checkout and the requested fresh build. Clock/EMC permission failures are recorded as unavailable, never changed silently. Failed outputs/logs remain outside Git; no full evaluation runs through ordinary CTest.
