# Fixed-token decode performance comparison

**Same input sequence, independently measured runs.** The README GIF replays actual completion timestamps for Block, One-shot and LeanStream processing the same fixed prompt and continuation on Jetson AGX Orin. The text is an author-written reference fed as model input, **not the models' independently generated answers**.

## Fixed workload

[Input text](fixed_decode_input.json) was declared before measurement. The pinned compatibility tokenizer encodes the raw prompt with BOS and the continuation without BOS. There is no chat template, truncation or EOS in the continuation. All three methods process exactly 112 continuation tokens after the same 23-token prompt.

> Explain how prediction-guided SSD weight streaming can help a memory-limited computer run a large language model.
>
> Answer:

The full reference text, exact prompt/continuation IDs, decoded prefixes, each run's processed IDs, predicted next-token IDs, completion timestamps and measured metrics are in [the measurement trace](fixed_token_decode_trace.json). Predictions may differ between methods, including EOS predictions, but are recorded only and never fed back. The fixed number of forwards is the workload; this does not evaluate free-generation quality or accuracy.

## Execution and timing

The three methods execute sequentially in independent processes, using the original public implementations, model, packed weights, predictors and policies. Each step:

1. Feeds the next fixed reference token with the preceding KV cache.
2. Executes the actual model forward, including attention, CPU prediction, SSD weight management/reads and sparse MLP GPU kernels.
3. Updates the KV cache and evaluates the ordinary `logits.argmax().item()` completion boundary.
4. Records a monotonic `perf_counter_ns` timestamp and reveals the reference input token whose forward has completed.

No reference token is displayed before its forward completes. At decode time zero the reference panes are empty. The predicted next token is not used to choose the next input. There is no extra per-step CUDA synchronization, profiling, artificial delay or synthetic timestamp interpolation.

Initialization, compilation, four-call warmup, cold native binding, runtime reset, prefill and final maintenance drain are outside decode timing. Prefill completion is established by the same host-visible logits boundary before the shared decode-time origin. Measured milliseconds are decode start through the last forward's completion; latency is that interval / 112, throughput is 112 / interval in seconds. Single-run figures are specific to this teacher-forced workload; they are not pooled free-generation benchmark results or statistical regression evidence.

Block and One-shot retain their original immutable 2,437-neuron hot sets, 2,867-neuron selections, individual-neuron storage and QD24 O_DIRECT reads. Their prediction timing differs by design. LeanStream retains Group2 active1434/keep1219, REPAIRED_KNEE, P0/P1-lite/P2, QD24, unsplit Direct-GQA and Progressive native pre-enqueue. Actual native bindings are checked across all 34 layers: A610/B32/reduction128, four warps per CTA. No synthetic LUT or online profiling is used.

The capture verifies identical processed IDs and decode counts. Baseline counters must contain 34 × 112 layer calls, unchanged fixed-hot membership and positive real SSD reads. Provenance includes public runtime module origins, native library hashes, pinned model/config identity, packed/predictor manifest hashes, hardware and software. Detailed launch commands and manifests remain in the isolated evaluation output directory.

## Measured fixed-token results

One independent run per method, all 112 steps:

| Method | Decode wall interval | ms/input token | Input tokens/s |
| --- | ---: | ---: | ---: |
| block | 44.359066 s | 396.063 | 2.525 |
| one-shot | 35.063922 s | 313.071 | 3.194 |
| leanstream | 14.900483 s | 133.040 | 7.517 |

These are actual timestamps through the last completion, not rates interpolated from historical results. Prompt IDs SHA256: `21ef4684b995dfe3ae57785ae21f2041e4fbf8893ff7a81f4a590c6bace19a31`; continuation IDs SHA256: `6468bb051c04d65df7487da8eafa092bfc6fe0964507d5ab6306224d43c61b1c`. The per-step intervals are retained without smoothing or outlier removal.

## Read the GIF

[Animation](speed_comparison.gif), [midpoint](fixed_token_decode_preview.png) and [final frame](fixed_token_decode_final.png) use three equal terminal windows. Each reveals the same tokenizer-decoded reference prefix at its own measured completion timestamps. All windows share a decode-time origin and a single replay scale, including the final hold. Text wraps and scrolls; tokens are never edited or animated character by character. There are no progress bars. Throughput labels come only from the measured fixed-token trace.

Historical matched free-generation throughput remains separate: Block 2.831, One-shot 3.281, LeanStream 8.877 tokens/s. Those historical values do not drive this animation or its labels.

## Capture and regenerate

See [the reproduction guide](../reproduce_ministral3_8b.md) for the pinned model, artifact availability and native build. Public artifact downloads remain unavailable until the complete checksummed collection is published. This capture requires no CoQA data.

```sh
python3 -m examples.ministral.reproduce.capture_fixed_tokens \
  --assets /data/leanstream-assets --model /data/ministral-model --build /data/leanstream-build \
  --input-file docs/media/fixed_decode_input.json --output /data/results/fixed-decode
```

Use a new output directory. Each method runs once in a separate process; the capture never changes the original method implementations. A custom input file must have `prompt_text` and `continuation_text` strings, produce 100–128 non-EOS continuation tokens, and round-trip through the pinned tokenizer. Incompatible inputs fail instead of silently truncating or substituting tokens.

Render the checked-in measured trace without inference or model downloads:

```sh
python3 tools/make_speed_gif.py \
  --trace docs/media/fixed_token_decode_trace.json \
  --output docs/media/speed_comparison.gif \
  --preview docs/media/fixed_token_decode_preview.png
```

Optional `--tokenizer /data/leanstream-assets/metadata/tokenizer_compat.json` additionally checks the tokenizer file hash, exact input encodings and every reference prefix. The renderer rejects unequal processed IDs, missing steps, nonmonotonic timestamps, inconsistent rates and incorrect native geometry. Pillow and DejaVu Sans Mono are optional presentation dependencies; ordinary toolkit builds do not import them. Output is 1600×700, approximately 14 seconds at 12 fps and deterministic with the same fonts/dependencies.

## Retained quality diagnostics

[Free-generation checks](diagnostics/README.md) retain the original open-ended trace, both predeclared story prompts, their real outputs and independent One-shot numerical checks. The [diagnostic animation](diagnostics/one_shot_diagnostic.gif) remains available for analysis but is not the README performance demonstration. Teacher forcing does not fix, conceal or measure the documented One-shot free-generation failures. `token_generation_trace.json` and its preview images retain the library diagnostic; `speed_trace.json` is separate historical pooled evidence. None supplies input to the fixed-token replay.
