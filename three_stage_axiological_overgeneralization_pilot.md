# Three-Stage Pilot

## Stage 1 — Validate the measurement

Start with `Qwen/Qwen3-8B` in non-thinking mode.

- Use binary choices with direct logits/logprobs.
- Sweep the competing cost across roughly logarithmic levels.
- Reverse A/B order and use several paraphrases.
- Estimate and plot \(P(	ext{target-value choice} \mid C)\).
- Check monotonicity, order robustness, paraphrase stability, and the estimated 50% threshold \(C^*\).

Goal: determine whether the tradeoff curve is a stable measurement instrument.

## Stage 2 — Validate sensitivity with prompting

Run the same benchmark on:

- `Qwen/Qwen3-8B`
- `google/gemma-3-12b-it`
- `allenai/Olmo-3-7B-Instruct`

Compare neutral prompting against a single-value prompt that explicitly treats the target good as intrinsically valuable.

Goal: verify that the instrument detects a known intervention and that the effect generalizes across model families.

## Stage 3 — Test causal change from training

Return to one primary model, initially `Qwen/Qwen3-8B`, and perform matched SFT on a single intrinsic value.

- Keep training examples benign and avoid severe value conflicts.
- Evaluate on held-out value-conflict curves.
- Measure the shift in \(C^*\), ideally as
  \[
  \Delta = \log\frac{C^*_{\text{post}}}{C^*_{\text{base}}}.
  \]
- Test whether the shift is value-specific, robust across scenario families, and not explained by generic fine-tuning drift or persona/extremity effects.

Goal: determine whether single-value training causes systematic OOD displacement of revealed tradeoff thresholds.
