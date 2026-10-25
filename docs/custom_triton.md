# Custom Triton computation with LeanStream

A streaming MLP is a coordinated **resident A / persistent B / reduction** bundle. A single arbitrary JIT function is not automatically a compute backend. This optional kit specializes the existing Group2 signed-INT4 bundle while retaining the toolkit's READY/claim/epoch machinery. Existing production kernel source and native enqueue are unchanged.

## Materialize a complete editable bundle

```sh
python3 templates/triton/materialize.py --output /data/my_bundle
python3 templates/triton/validate.py /data/my_bundle
PYTHONPATH="$PWD:/data" python3 examples/custom_pytorch/run.py \
  --mode validate --bundle my_bundle \
  --library /data/leanstream-torch-build/libleanstream_torch.so \
  --output /data/new-custom-results
```

The generator emits `group_arithmetic.py`, `resident.py`, `persistent.py`, `reduction.py` and a package initializer. Persistent and reduction function bodies come directly from public source. Resident arithmetic is factored verbatim into `_resident_arithmetic`; its epoch wait, inactive zeroing and finish remain framework-owned. The validator reconstructs these framework functions from public source and rejects protocol changes. There is no function injection into an already compiled Triton kernel: this is ordinary Python source import and cold Triton specialization of a complete bundle.

Edit the generated **arithmetic** module: `weight` for packed interpretation/dequantization, `_resident_arithmetic` for resident work and `_compute_pair` for streaming work. Coordinate changes to both A and B. An activation change in B alone does not change A and is incorrect. Keep group granularity, signatures, descriptor consumption and partial layout compatible, or provide a separately audited optional adapter. Do not edit READY/claim logic merely to support new arithmetic.

## Packing, arithmetic and numerical example

Supported input is one FP16 width-H vector. Each logical group contains two neurons; each neuron has gate/up/down signed-INT4 rows. Low nibble is the even coordinate, high nibble the odd coordinate. Each projection occupies H/2 bytes; each group occupies3*H bytes. FP16 scale lookup is physical-neuron/projection/coordinate-block, per128 coordinates. The loader selects Cache or Staging by source_kind, source_slot selects packed data, and physical_group selects scales. Logical group identity does not imply file order.

For every neuron the original arithmetic dequantizes weight to FP16 then FP32, computes gate and up, rounds each to FP16, rounds SiLU(gate) to FP16, multiplies by up and rounds again, then forms FP32 down-projection partials. Reduction rounds the final output to FP16. Do not accidentally replace these boundaries with different quantization or activation arithmetic.

At H=512, INT4=1, FP16 scale=.05 and FP16 input=.1, a gate is approximately2.56 before activation. The exact reference must use the stated half-rounding boundaries rather than decimal approximations. `examples/custom_pytorch/run.py::reference` independently dequantizes and evaluates selected groups with those boundaries. Validation compares actual model output with absolute tolerance.002, without changing the existing production tolerance.

A concrete editable specialization doubles decoded weights in both A and B:

```sh
python3 templates/triton/materialize.py --output /data/double_bundle --weight-multiplier 2
PYTHONPATH="$PWD:/data" python3 examples/custom_pytorch/run.py \
  --mode validate --bundle double_bundle --weight-multiplier 2 \
  --library /data/leanstream-torch-build/libleanstream_torch.so \
  --output /data/new-double-results
```

This deliberately changes only the synthetic example's interpretation, with a matching independent reference. It is not an optimization, trained model, or modification to production kernels. It demonstrates real arithmetic specialization of a full bundle through WeightManager, SSD I/O and GPU persistent consumption.

## Descriptors and output ownership

ExecDesc is five uint32 fields (20bytes):

| Field | Meaning |
| --- | --- |
| logical_group | Stable predictor/application ID |
| physical_group | Weight/scale ordering ID |
| source_slot | Cache or staging slot containing valid bytes |
| dest_slot | Deterministic logical activation destination |
| source_kind | 0 Cache, 1 Staging |

Core descriptors are immutable mapped A metadata. Noncore finalized descriptors belong to DeviceExecQueue. `ready_indices` identifies descriptor indices that are safe to read, not logical IDs. The C++ producer writes descriptor/data first, then releases ready_count. GPU uses system-scope acquire before accessing the published prefix. QueueControl epoch is at uint32 index3, total_count at2, ready_count at0 and error at5. It occupies128bytes; the queue's claim_count cacheline is separate. The production Triton consumer uses its own three-int GPU Claim buffer, with a separately incremented CLAIM_EPOCH; do not substitute QueueControl.claim_count for it.

The included kernel indexes activation H by `dest_slot*2 + neuron_in_group`. Resident A stores one partial per four neurons; persistent B stores **one partial per B descriptor index**, offset by A program count. It does not index its partials directly by dest_slot. Compact reduction understands this layout and folds B partial pairs. Unused suffix partials are zeroed before reduction. Changing this convention in one stage creates races, stale contributions or wrong sums.

Runtime owns final membership, descriptor construction and publication. The unchanged persistent kernel waits for the correct epoch, observes ready_count, claims each descriptor index exactly once and chooses cache/staging loads. Discarded speculative reads are never in final descriptors and must not execute. Cache/staging slots cannot be reused until GPU completion and all transport callbacks retire. Errors invalidate output even if reduction was submitted.

## Native cold binding

`Group2Bundle` in `leanstream/integrations/torch/compute.py` is the executable binding template; `templates/triton/bind.py` points to it. It allocates stable buffers, warms an empty finalized B plan, compiles A/B/reduction outside decode, then packs handles with existing `backends.triton.binding.pack_kernel`. The optional bridge calls existing `TritonBackend::bind` and `enqueue`.

For the toy H512, eight-group example with two core groups:

```python
# Complete implementation: Group2Bundle._prepare
# pa=1, pb=6; B specialization WORKERS=32, G=2, WIDTH=512.
# Python compiles each kernel, then:
args, epoch_index, claim_index, names = pack_kernel(compiled_b, values)
# The bridge forwards candidate3/stage1, CUfunction, grid32, warps4,
# shared-memory bytes, args/count, EPOCH index and CLAIM_EPOCH index.
```

Inputs/outputs, scales, packed cache, mapped staging, core/queue descriptors, ready indices, control, Claim, partials, completion buffers and kernel objects remain owned through retirement. The native path substitutes only runtime epochs and launches A->B->reduction on the supplied CUDA stream. It does not compile, benchmark, profile, allocate argument storage, or search candidates during decode.

Existing binding accepts at most **32 runtime pointer/integer arguments**, integer scalar types i32/u32/i64/u64, a non-specialized EPOCH and optional CLAIM_EPOCH. Float scalars must be compile-time constants or represented through a supported buffer. Clusters (`num_ctas!=1`) are unsupported. Native B grids are restricted to existing4/8/16/32-CTA candidates, warps4; this adapter binds only candidate3/32. A/reduction grids follow geometry. WORKERS must equal the compiled B grid. A custom parameter list exceeding these limits requires a proposed compatible optional adapter; this kit does not silently change the native ABI.

The hot path uses the validated fallback without a LUT. Generic offline LUT APIs remain available in the existing toolkit, but this first Python attachment does not expose a new LUT configuration surface. Synthetic example tables are never loaded.

## Stream and lifecycle contract

The model continues its normal stream. Input copy is GPU-to-GPU into a stable cold-bound buffer. Progressive pre-enqueues B before final membership; CPU prediction must therefore not wait on work queued behind B. Producers on other streams need correct ordering and sufficient resources; do not assume waiting persistent CTAs leave arbitrary producer kernels runnable. See [prediction readiness](integrating_pytorch.md#activation-readiness-is-an-explicit-user-contract).

A timing-disabled event after reduction proves GPU completion. Background commit retains fixed residency and calls Runtime retirement only after that event. Retirement stops admissions and drains callbacks. The next use of the same layer waits for the preceding retirement; there is no universal cross-stream/reentrant support. Cancellation publishes error; partial enqueue requires completion proof before any borrowed tensor or mapped buffer is released.

The platform must provide integrated coherent mapped CUDA memory, host-native atomics and system-scope release/acquire. The native allocator checks those properties. This adapter additionally restricts to SM87. File extents and staging are512-byte aligned; H must be a multiple512. Staging is fixed logical storage for every group, not bounded dynamic staging. Arbitrary discrete GPU transport, arbitrary group sizes/formats and low-memory checkpoint loading are unsupported here.

## Correctness and troubleshooting

Run `run.py --mode validate` for real cached/staging descriptors, SSD misses, final-only exactly-once B execution, reduction and repeated epochs. The optional CPU test validates mapping/budgets/model replacement and rejects edited framework protocol bodies. `tests/torch/gpu_progressive.py` uses a clearly labeled controlled-producer gate to prove early I/O, attention progress while I/O is held, three generations, non-preemptive started reads and final-only consumption. `gpu_errors.py` tests cancellation after live submission, transport retirement and fresh-epoch recovery. These are correctness tests, not benchmarks.

- A hang before epoch publication usually means incorrect finalize ordering or a CPU predictor waiting on the blocked CUDA stream.
- Duplicated/missing groups suggest a mismatched claim generation, descriptor index or illegally edited consumer protocol.
- Wrong output with correct counts suggests physical-scale mapping, quantization, A/B arithmetic mismatch or partial/reduction indexing.
- Stale output on the next token suggests missing retirement or reused active buffers.
- Unsupported parameter/grid errors require a matching cold specialization, not trial launches.
- A fatal CUDA error may prevent completion proof. The integration retains owners instead of freeing live pointers; restart that evaluation process.

Supported: the exact Group2 coordinated template and documented signature/geometry. Experimental: application-specific asynchronous feature publication and Attention Mid. Unsupported: arbitrary JIT injection, arbitrary quantization/group sizes, cluster launches, concurrent/reentrant decode and arbitrary-platform coherence claims.
