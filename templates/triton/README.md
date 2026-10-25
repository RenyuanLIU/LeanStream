# Editable Group2 Triton bundle

These are cold source-generation tools, not dynamic JIT function injection. The executable template is materialized from the existing public kernel bodies:

```sh
python3 templates/triton/materialize.py --output /data/my_bundle
python3 templates/triton/validate.py /data/my_bundle
PYTHONPATH="$PWD:/data" python3 examples/custom_pytorch/run.py --mode validate --library /data/ls-build/libleanstream_torch.so --bundle my_bundle --output /data/custom-check
```

Generated files are `group_arithmetic.py`, `resident.py`, `persistent.py`, `reduction.py`, and `__init__.py`. Edit `weight` and `_compute_pair` in `group_arithmetic.py` for compatible packing/dequantization/group-local arithmetic. Resident arithmetic is factored into `_resident_arithmetic` in that module and uses the same edited `weight`; persistent calls the edited pair arithmetic. Coordinate activation/math changes in both arithmetic functions. Keep resident/persistent/reduction framework functions identical to the generated template; `validate.py` checks them. Framework READY/CLAIM/epoch logic is copied directly from the public source, never independently implemented.

A runnable arithmetic specialization uses `--weight-multiplier 2` when materializing, then adds `--weight-multiplier 2` to the toy numerical command. This intentionally changes the synthetic weight interpretation, not production code. It tests A/B/reduction together through the real Runtime and SSD. `bind.py` points to the single shared cold-binding implementation.

This first bundle supports Group2 signed INT4, FP16 per128 scales, width multiple of512, and four warps/32 persistent CTAs. Arbitrary group sizes, quantization formats, kernel signatures, or reductions need an additional audited compute adapter. Merely editing a grid or passing a JIT callback is not supported. See [the complete guide](../../docs/custom_triton.md).
