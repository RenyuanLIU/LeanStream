# Generation quality and token provenance

The traces in this directory are unedited generation records, not answer-quality benchmarks. The model is the pinned Ministral-3 8B checkpoint and all methods use the original greedy inference settings. The default open-ended SSD explanation prompt and two predeclared, author-written story/question prompts probe different input styles. Both story prompts are run through all three methods, in the order recorded in `prompt_plan.json`.

## Original open-ended prompt

[Original trace](original_token_generation_trace.json) preserves all token IDs, completion timestamps and decoded prefixes. Its SHA256 is `eacb05db57c7b4c1a244a8ecc66d0098750afab9ee3a2be5e5d5a44ac74fac59`. Every prefix was independently decoded using the validated tokenizer, SHA256 `b8e02db616fe020ed1bd7a9a8eb4db73a9693012909c75f9ba2c6f8abd26fadb`. One-shot's repeated “helps a small computer” is present in the generated IDs. It is not introduced by GIF rendering, token-fragment concatenation or tokenizer reconstruction. The other methods' answers are also preserved; their factual accuracy is not asserted.

## What One-shot computes

The historical One-shot baseline uses 512 predictor features from the raw pre-attention layer input. It selects 2,867 individual neurons, prefetches missing weights, waits at the MLP, and computes with that selection. There is no final-input selection correction. Its 2,437 fixed-hot entries remain immutable. Block instead predicts from the actual normalized MLP input using 1,638 features. These methods therefore differ in approximation as well as scheduling.

The independent numerical reference checks the arithmetic for the **selected** neurons. It checks unique descriptor membership, Cache/Staging source selection, exact packed bytes against SSD storage, independent dequantization and PyTorch gate/up/SiLU/down computation. The original tolerances are unchanged: `rtol=0.02`, `atol=0.002`. Passing this reference does not establish that early prediction selected the best neurons, that sparse execution matches the dense model, or that arbitrary prompts retain language quality.

The historical-prompt smoke passed all 68 layer comparisons (two decode calls across 34 layers), with finite outputs, unchanged fixed-hot IDs, and 797,528,064 actual SSD bytes read. No runtime, predictor, baseline, kernel or numerical tolerance was changed for these checks.

The original open-ended prompt also passed **918 layer comparisons**: four warmup calls and 23 measured-generation calls across all 34 layers. Its first 24 generated IDs exactly matched the original trace, including “helps helps” and the repeated small-computer phrase. The immutable hot set and real SSD reads were verified. [Numerical check metadata](one_shot_numerical_check.json) records the source/native hashes, original tolerances, audit hash and reproduced prefix. Reference-check timing is excluded from every presentation trace and performance claim.

This is a reproducible **approximate-inference generation-quality failure** on this prompt. No selected-neuron arithmetic, descriptor/source, packed-weight or tokenizer error was found in the checked calls. The behavior is consistent with the original early, reduced-feature selection having no final correction. These bounded checks do not prove that every possible implementation error is absent, or establish dense-model accuracy. The baseline is preserved unchanged; a readable grounded example must not be interpreted as a general quality guarantee.

## Predeclared grounded prompts

Both prompts used the same pinned model/tokenizer, original predictors and configurations, greedy decoding, seed 42, EOS 2 and a maximum of 64 new tokens. Within each prompt all three methods received identical IDs and ran in separate sequential processes. No additional candidates were tried.

### Garden

```text
Read the story and answer the question using only the story.

Story: Maya planted carrots and beans in her garden. Every morning she watered the plants before school. On Saturday, she picked the ripe carrots and made soup for her family.

Question: What did Maya do with the carrots she picked?
Answer:
```

**block**, 5 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 She made soup.
```

**one-shot**, 2 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 She
```

**leanstream**, 5 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 She made soup.
```

### Library

```text
Read the story and answer the question using only the story.

Story: Leo borrowed a book about planets from the town library. He read it with his sister on Sunday. On Monday, he returned the book and borrowed one about birds.

Question: What was the second book about?
Answer:
```

**block**, 8 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 The second book was about birds.
```

**one-shot**, 48 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 The second book was about the second book was about the second book about the second book about the second book about the second book about the second book about the second book about the second book about the second book about the second book about the
```

**leanstream**, 8 tokens including the prefill-selected token; EOS: yes. Exact decoded output (the initial space is retained):

```text
 The second book was about birds.
```

No candidate met the predeclared readability condition. Garden One-shot terminates after “She”; library One-shot repeats a phrase. Both are failures to answer the question. The library trace is used only as an explicitly labeled diagnostic replay because it makes the repetition visible alongside the complete answers. It is not a quality-qualified replacement example or a throughput comparison, and does not resolve the original One-shot limitation. No generated tokens were corrected or discarded.

The retained [diagnostic animation](one_shot_diagnostic.gif) is separate from the README's fixed-token performance comparison. Regenerate it from the library trace using `tools/make_speed_gif.py --trace docs/media/diagnostics/library_token_generation_trace.json --output docs/media/diagnostics/one_shot_diagnostic.gif --diagnostic`. Teacher-forced performance measurements do not resolve or measure these free-generation quality failures.
