# Genuine autoregressive generation demonstration

The README GIF shows Block, the recalibrated default One-shot, and LeanStream on
the same author-written Library prompt. The original capture called One-shot an
experimental recalibrated variant; those exact1638-feature parameter binaries and
early-normalization/pre-attention scheduling are now the v2 public default.
No token sequence or timestamp was changed when adopting it as the default.

[GIF](library_visit.gif) · [Midpoint](library_visit_gif_midpoint.png) ·
[Final frame](library_visit_gif_final.png) · [Original measured trace](library_visit.trace.json)

## Prompt and provenance

The exact raw prompt is in [library_prompt.txt](library_prompt.txt); no chat template
is applied. Model: cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit at
`e3a799844ba08f27fddbf118e154057eb0105889`. The compatibility tokenizer SHA256 is
`b8e02db616fe020ed1bd7a9a8eb4db73a9693012909c75f9ba2c6f8abd26fadb`.
The trace contains exact prompt IDs, output IDs, actual feedback IDs, decoded
prefixes, completion timestamps, model/config/native-source identities and EOS.
Historical predictor training-manifest identity:
`6c057be64a2aef3d11646d7396a5d4dcbc3d27b830c5ddc1d4dc9c71ea6cb675`.
The public v2 runtime manifest changes notices/packaging, while every parameter
binary retains its original training-manifest hash. Catalog validation enforces it.

Three independent greedy runs used max64 output tokens and EOS2. Prefill produces
the first token, shown at t=0 and excluded from decode throughput. Subsequent tokens
appear only at the ordinary logits.argmax().item() completion boundary. Actual
predictions are fed back with KV cache. No teacher forcing, substituted text,
smoothed timing or per-method playback scaling. All panes share one linear decode
time scale plus a final hold. Block reaches64 tokens; One-shot57 and LeanStream56
reach EOS. Their unequal outputs, invented details and awkward wording are retained.
This single-prompt timing is separate from the matched pooled benchmark.

The six medium-length prompts were declared before their outputs. This Library
prompt was selected for a readable One-shot answer without a serious loop. The
method still failed other prompts (Sandwich degeneration and Seed incoherence),
and LeanStream also looped on several probes. This animation is not a universal
quality claim. Historical diagnostic tools/data remain under ../diagnostics/.

## Capture and render with the public default

```sh
python3 -m examples.ministral.reproduce.capture_tokens \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --prompt-file docs/media/autoregressive/library_prompt.txt --max-new-tokens 64 \
  --output /data/results/autoregressive
python3 tools/make_autoregressive_gif.py \
  --trace /data/results/autoregressive/token_generation_trace.json \
  --tokenizer /data/leanstream-assets/metadata/tokenizer_compat.json \
  --output /data/results/autoregressive/comparison.gif
```

No experimental variant flag or predictor training is needed. Timing-dependent
speculative I/O may change actual generated answers; do not require bitwise-equal
trajectories. To reproduce the saved rendering without running inference:

```sh
python3 tools/make_autoregressive_gif.py \
  --trace docs/media/autoregressive/library_visit.trace.json \
  --output /data/results/library_visit.gif
```

Pillow and DejaVu Sans Mono are optional renderer dependencies. Exact tokenizer
checking additionally uses tokenizers. Rendering is deterministic with matching
dependencies/fonts. File identities are in SHA256SUMS. The saved GIF SHA256 is
`7335a1d78fb35a4cb693791264c3b08ac5b2880ac96dc7b374b1350487e8fd33`.
