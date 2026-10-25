# Integrating LeanStream with a PyTorch model

**User owns model-specific computation. LeanStream owns asynchronous execution coordination.** This optional integration keeps your model's attention, normalization, residuals, KV cache and outer forward. It replaces configured MLP modules and delegates I/O and READY publication to the existing C++ Runtime. It does not modify the validated Ministral application.

## Scope and audit

`Runtime::begin_layer` creates a LayerStep; `update_prediction` refines SSD priorities; `enqueue_compute` submits a borrowed ComputeBackend; `finalize` freezes membership; `commit` retires after GPU completion. `LayerStorage` borrows GroupLayout, cache bindings, staging, I/O states, core descriptors and DeviceExecQueue. Its owner must retain them through GPU completion **and** transport callback retirement.

The existing `TritonBackend::view()` is compatible with Runtime, but there was no generic Python Runtime lifecycle ABI. The optional `integrations/torch/native/bridge.cpp` wraps these existing classes. It owns mapped metadata and O_DIRECT file handles, uses the original STREAM_IO engine, and delegates native A/B/reduction launch to the unchanged TritonBackend. A timing-disabled event after reduction drives background retirement via `ComputeBackend::defer_commit`. Neither predictor callbacks nor storage completion launch GPU work.

The first supported compute adapter is deliberately narrow: Group2 signed INT4, SiLU gated MLP without bias, FP16 output, per128 physical scales, WIDTH a multiple of512, an even nonempty resident core, one CUDA stream, and batch1 single-token decode. Native mapping rejects devices lacking integrated coherent memory and host-native atomics; Python additionally restricts this tested adapter to Jetson SM87. Generic Runtime supports other compute backends, but this attachment does not pretend to support them automatically.

Ministral's calibrated predictors, P0/P1/P2 sizes, cache policy, raw residual publication, and exact projection/norm orchestration are application-specific. They are not copied into this integration. Existing production source remains unchanged.

## Build and run a complete example

Use the existing supported Jetson environment: CUDA11.8, vendor PyTorch2.1.0a0, Triton3.1.0, NumPy1.24.4. C++17, pthreads and CUDA development libraries are required. Transformers is needed only for the optional HF recipe; Pillow only for the GIF generator. No automatic installation or download occurs. Generic CMake targets are unchanged and require none of these optional Python packages.

From the project root:

```sh
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
python3 integrations/torch/build.py --output /data/leanstream-torch-build
# Optional existing CPU predictor shared library:
python3 integrations/torch/build.py --predictor --output /data/leanstream-torch-build
python3 examples/custom_pytorch/run.py --mode inspect
python3 examples/custom_pytorch/run.py --mode validate \
  --library /data/leanstream-torch-build/libleanstream_torch.so \
  --output /data/toy-validation
```

Use a **new output directory on an O_DIRECT-capable local SSD**. The example creates small synthetic files, a nonidentity group permutation, a deterministic predictor, a tiny gated MLP and a decoder layer. It runs dense prefill, replaces only the MLP, performs real SSD reads with public WeightManager, consumes Cache and Staging descriptors through public Triton persistent B, reduces and returns a Tensor into the original forward. It checks independent arithmetic, exactly-once finalized B execution, repeated epochs, detach and original-module restoration. There is no model download or predictor training. `--mode basic` and `--mode progressive` run separately.

## Manual attachment

This complete small-model code uses the same objects as the executable example:

```python
from pathlib import Path
import torch
from examples.custom_pytorch.run import ToyModel, pack, predictor
from leanstream.integrations.torch import (
    attach_leanstream, ModelPoints, ComputeSpec, CachePolicy,
    PredictionSchedule, PredictionStage,
)

weights, _ = pack(Path('/data/new-toy-weights'))
model = ToyModel().cuda().half().eval()
points = ModelPoints(layers='layers', mlp='mlp', attention='self_attn',
                     pre_mlp_norm='post_attention_layernorm')
schedule = PredictionSchedule(
    early=PredictionStage(predictor, budget=4, io_budget=4),
    final=PredictionStage(predictor, budget=5, io_budget=5),
)
with torch.inference_mode(), attach_leanstream(
    model, points=points, weights=[weights],
    compute=ComputeSpec(Path('/data/leanstream-torch-build/libleanstream_torch.so')),
    cache_policy=[CachePolicy(resident=(0, 1, 2), core=(0, 1))],
    prediction=schedule, mode='progressive',
) as handle:
    y = model(torch.ones(1, 1, 512, device='cuda', dtype=torch.float16))
    handle.drain()
```

`attach_leanstream` returns a **lifecycle handle**, not a replacement model. Continue calling the original model. `detach()`/`close()` restores original modules and removes hooks. A delegating layer-forward guard handles exceptions on vendor PyTorch builds lacking always-call hooks; it does not replace the layer's mathematical implementation. Double attachment is rejected. Use the handle as a context manager. Predictors and external feature-publication workers remain user-owned; close/join them only after the handle drains and their own input publications retire. Explicit `handle.reset()` aborts any unfinished step and drains; the application still owns its KV cache and prompt resets.

Supply one WeightSpec and CachePolicy for each configured decoder layer. Attention and normalization paths are validated; final prediction receives the actual MLP argument, rather than assuming the norm can be reconstructed. Shared/reentrant modules, concurrent forwards, batches>1, arbitrary input layouts, autograd, training, torch.compile/CUDA graph capture and concurrent stream changes are unsupported. A layer must receive hidden states as its first argument or `hidden_states=`. Unknown forward semantics require manual inspection.

## Inspection and automatic discovery

```python
from leanstream.integrations.torch import inspect_model, ModelPoints
report = inspect_model(model)  # read-only; no hooks or compilation
print(report)
# Only after reviewing paths, geometry, activation and model semantics:
points = ModelPoints.auto()
```

Discovery checks `model.layers`, `model.model.layers` and `layers`, gated projection dimensions, no MLP bias, attention presence and norm width. Multiple matching layer containers are rejected. It returns paths, dimensions, reasons and ambiguities. This is a structural inspection, **not proof of arithmetic, attention signatures, predictor compatibility or model support**. Manual paths override discovery. No FX compiler or automatic checkpoint conversion is used.

## Prediction and execution modes

A predictor is `predictor(source, layer_id, activation, budget) -> ranked_logical_ids`, or a Future that is already ready when needed. IDs must be Python integers, unique, in range, priority ordered and within budget. Duplicates are rejected. `budget` counts ranked groups; `io_budget` limits new EMPTY admissions, not cached/READY/INFLIGHT reuse. Group identity must match WeightSpec, independently of physical file ordering.

**Basic** starts at the actual MLP input, predicts, updates priorities, finalizes, explicitly waits for required I/O, then enqueues sparse compute. It is a blocking correctness/integration mode. Set `early=None` and `mode='basic'`.

**Progressive** begins at the decoder layer-input hook, publishes early priorities, lets the original attention/norm execute, copies the final GPU input into the cold-bound input buffer and pre-enqueues A/B/reduction. Final prediction then refines priorities and freezes membership. B is already submitted and consumes only finalized READY descriptors. `handle.refine(layer, ids, io_budget)` supports further application-owned refinements during an active step. Every final selection must include the configured core. Speculative reads do not authorize speculative computation.

New generations reorder only unstarted reads; inflight I/O remains non-preemptive, READY results may be reused, and finalized membership is immutable. A reference fixed-residency policy is provided; it is not REPAIRED_KNEE or an adaptive cache. Admission/retention policy remains replaceable at the application level. This first Python bridge supports fixed resident/core sets only; dynamic cache installation needs a separate optional policy adapter.

### Activation readiness is an explicit user contract

The integration **does not implement a generic GPU-to-CPU feature publication engine**. Synthetic tests use metadata-only deterministic prediction, with no activation transfer. A real CPU predictor must consume an already CPU-ready representation or application-owned asynchronous publication. The wrapper does not call `.cpu()`, `.item()` on activations, or wait on an unfinished Future to get a newer prediction. An unfinished early Future is polled opportunistically; an unfinished final Future is rejected.

In Progressive, final predictor code runs after persistent B has been submitted. **Never enqueue a feature extraction/copy behind B on that same stream and synchronously wait for it:** B is waiting for final membership. Prepare CPU features before submission or publish them asynchronously on a correctly ordered producer stream. Retain publication buffers and activation owners until prediction retires. Unmodified arbitrary Python CPU predictor code that synchronously copies the supplied CUDA activation is not a supported Progressive predictor. Basic can use an explicitly blocking application predictor, with the cost understood.

`NativeCPUPredictor` adapts the existing `ls_create/ls_predict/ls_destroy` ABI with user-provided compatible parameters and an `input_provider` returning `(CPU_features, input_scale)`. It never trains, gathers or quantizes implicitly. Supply stage-specific feature indices/preprocessing, native weight layout, stride multiple32, precision and fixed budget; use separate contexts for different stages/concurrent callers. See `include/leanstream/predictor.h`. The reference implementation sets the OpenMP team size and pins its workers. Set `torch.set_num_threads(threads)` explicitly before constructing this adapter; mismatched Torch/OpenMP team sizes are rejected, rather than silently altering Torch settings. Parameters trained for Ministral cannot be assumed valid on Qwen.

## Weight and cache contract

`WeightSpec(path, width, layout, scales)` uses the public GroupLayout fields: logical_group, physical_group, file_offset, bytes. The supplied packer is only a synthetic tutorial. Logical and physical IDs must each be permutations; offsets must be 512-byte aligned, nonoverlapping and within the file. Every group is two neurons, gate/up/down low-nibble-first signed INT4, each row WIDTH/2 bytes. Group bytes=3*WIDTH. Scales are FP16 `[physical_group, 2, 3, WIDTH/128]`.

Predictors return logical IDs. Descriptors map them to physical scales and cache/staging source slots. Fixed cache slot order follows `resident`; setup reads those bytes once. Generic staging uses one slot per logical group, stride=group_bytes. `CachePolicy` units are **groups**, not percentages or neurons. Physical cache bytes=resident_count*group_bytes; logical staging bytes=group_count*group_bytes, plus metadata/partials/scales. Strict bounded staging and strict physical85% enforcement are not implemented.

See [custom Triton](custom_triton.md) for exact descriptor and partial-result mapping. Users with other quantization/group layouts must supply a compatible complete backend adapter; WeightManager itself remains unaware of quantization.

## Prefill, generate and memory

The initial integration preserves original dense MLPs for multi-token prefill and streams only `[1,1,H]` decode. It retains dense weights and therefore **makes no low-memory checkpoint-loading or total-memory-saving claim**. A single-token prompt is treated as sparse decode; automatic phase detection cannot distinguish single-token prefill. For HF generation use a normal multi-token prompt and inspect the model's forward/cache behavior first. General skip-loading/unloading dense weights is outside this kit.

No attention or KV-cache implementation is replaced. Dense prefill performs ordinary model initialization; decode begins a fresh native epoch per layer call. Each bundle is bound to its initialization CUDA stream, retains fixed input/output/descriptor buffers and compiled functions, and rejects a stream change. The returned result is cloned into ordinary PyTorch-owned storage on that stream. Call `drain()` at prompt/end-of-session boundaries; retirement waits occur on the background worker, with same-layer reuse waiting for the preceding epoch. No performance equivalence to the specialized Ministral orchestration is claimed.

On prediction/finalization/submission failure, cancellation publishes the queue error, establishes stream completion and drains transport callbacks before freeing buffers. Even partial enqueue can leave GPU work live. If completion cannot be proved after a CUDA failure, owners are retained in quarantine rather than freeing borrowed pointers. Restart the process after such a fatal device failure; output from a failed step is invalid.

## Qwen2.5-oriented walkthrough — DOCUMENTED ONLY for E2E

Start with your normal `AutoModelForCausalLM.from_pretrained(...)`. Keep that model. Qwen2.5 commonly exposes `model.layers`, `mlp`, `self_attn` and `post_attention_layernorm`; run read-only inspection instead of assuming a universal layout.

You must provide a matching Group2 packed bundle and logical mapping, physical FP16 scales, stage-specific trained/calibrated predictors or a custom predictor, fixed resident/core IDs, and numerical evidence for Qwen's activation/geometry. Use the built-in kernels only where its signed INT4/Group2/per128/SiLU contract is actually satisfied. A different layout needs the [editable complete bundle](custom_triton.md). There are no downloadable Qwen predictors or packed weights in this kit, and no tested Qwen E2E claim.

`examples/custom_pytorch/attach_model.py` is a complete opt-in HF manual recipe. It loads only a user-supplied local checkpoint, then inspects, attaches and calls `generate`. It does not download a model:

```sh
python3 examples/custom_pytorch/attach_model.py --model /data/qwen-local --inspect-only
python3 examples/custom_pytorch/attach_model.py --model /data/qwen-local \
  --manifest /data/qwen-stream/weights.json --predictor my_predictors:create \
  --library /data/leanstream-torch-build/libleanstream_torch.so \
  --input-ids /data/prompt_ids.json
```

The JSON manifest contains `layers` with `file`, `width`, `scales` (a CPU FP16 `.npy`), `layout` (GroupLayout objects), `resident`, `core`; top-level `early_budget`, `early_io_budget`, `final_budget`, `final_io_budget`. Paths are relative to the manifest. The factory returns `(early_predictor, final_predictor)` implementing the readiness contract. Prompt IDs must contain at least two tokens. These inputs are researcher responsibilities, not supplied validated Qwen assets.

## Attention Mid — EXPERIMENTAL / DOCUMENTED ONLY

Normal integration does not split attention. `PredictionSchedule(attention_mid=...)` rejects non-None values. The validated Ministral default remains unsplit Direct-GQA with its existing orchestration.

Head splitting publishes only selected heads; a partial output projection publishes a transformed contribution, not the final residual/MLP input. A KV-prefix checkpoint publishes attention over only a prefix of context, with different normalization from full attention. These are different activation distributions and may need separately calibrated predictors.

An application publisher must specify: exact partial value and normalization/residual transform; an owned mapped buffer; producer stream dependencies; system-release epoch after all writers finish; CPU acquire before feature reading; a predictor bound to that representation; ranked logical IDs published through `refine`; and independent final selection from the actual MLP input. The buffer stays live until producer and predictor retire. CPU/I/O completion still cannot launch GPU work. A producer stream must not be starved by waiting persistent CTAs.

Existing Ministral options (`head_split16`, `semantic_projection`, `semantic_mid_publish`, `checkpoint`, `midpoint`, `projection_overlap`, `semantic_mid_rescue`) are reference research paths, not interchangeable switches. Its checks require specific split/publication combinations, and REPAIRED_KNEE disallows Mid/relaxed variants. Do not bypass these restrictions. This kit documents the extension contract; it does not claim a working generic asynchronous Mid implementation.

## Support matrix

| Feature | Status / boundary |
| --- | --- |
| Manual PyTorch integration | FUNCTIONAL / TESTED, toy Jetson inference |
| Auto model discovery | FUNCTIONAL / TESTED, structural inspection only |
| Basic sparse MLP | FUNCTIONAL / TESTED, real SSD and GPU |
| Progressive streaming | FUNCTIONAL / TESTED, early/refined priorities and pre-enqueued READY consumption |
| Real SSD I/O | FUNCTIONAL / TESTED, O_DIRECT NVMe |
| Native Triton pre-enqueue | FUNCTIONAL / TESTED through unchanged native backend |
| Built-in Group2 INT4 kernels | VALIDATED existing backend; new toy integration tested separately |
| Custom Triton specialization | FUNCTIONAL / TESTED for compatible Group2 arithmetic template |
| Custom A/B/reduction bundle | FUNCTIONAL / TESTED for supplied signatures/layout/geometry; arbitrary bundles unsupported |
| Attention Mid | DOCUMENTED ONLY / EXPERIMENTAL |
| Qwen E2E | DOCUMENTED ONLY, no supplied compatible assets |
| Arbitrary group sizes / quantization | UNSUPPORTED by this Python adapter |
| Discrete GPU | UNSUPPORTED by mapped publication contract |
| Low-memory checkpoint loading | UNSUPPORTED |
| Strict bounded staging | UNSUPPORTED |

## Optional tests

```sh
LEANSTREAM_TEST_PREDICTOR=/data/leanstream-torch-build/libpredictor.so PYTHONDONTWRITEBYTECODE=1 python3 -B tests/torch/test_integration.py
python3 integrations/torch/build.py --testing --output /data/ls-testing
python3 tests/torch/gpu_errors.py --library /data/ls-testing/libleanstream_torch.so --output /data/new-error-assets
python3 tests/torch/gpu_progressive.py --library /data/ls-testing/libleanstream_torch.so --output /data/new-gate-assets
```

The testing build alone enables controlled producer gates and failed-submission injection; it is not for benchmarks. The concurrency test performs real pread after a controlled gate, proves attention completes while requests remain inflight, applies three priority generations, checks non-preemptive reads and final-only exactly-once execution. Its deliberate gate and validation synchronization are absent from normal integration. Generic tests remain separate; no full-model benchmark or CoQA evaluation is run here.
