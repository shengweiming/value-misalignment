# Research Log — 2026-08-26

## Repository workflow and H4rmony SFT planning

The local checkout was fast-forwarded to repository commit
`613693f414ddd105149d694ecc66b17bf7b5a02f`, which includes the August 25
H4rmoniousCaramel matched-checkpoint pilot. A repository-level `AGENTS.md` was
added to make two working conventions persistent for future sessions: completed
task changes should be committed unless the user explicitly opts out, and
substantive work should be recorded in the dated research log in the same commit.

The H4rmony project's public `data/train.csv` currently contains 517 SFT rows.
Its existing target outputs are short (about 123 characters on average), but the
planned R1-generated answers are not yet present in this repository, so training
time depends principally on whether “answer only” excludes the long reasoning
trace and on the resulting token-length distribution.

For one final-answer completion per H4rmony prompt, a dense 8B base model, LoRA
rank 16, three epochs, BF16, gradient checkpointing, sequence packing, and a
single A100, the provisional estimate is:

- roughly 10–25 minutes of optimization if examples fit comfortably within a
  1,024-token sequence length;
- roughly 20–45 minutes at a 2,048-token sequence length;
- roughly 45–90 minutes at a 4,096-token sequence length; and
- approximately 25–60 minutes end to end for the expected answer-only case once
  model download, tokenization, evaluation, and adapter saving are included.

These are planning ranges rather than measured benchmarks. Retaining long R1
reasoning traces could push the run to roughly 1–3 hours, and a cold Colab model
download or a slower A100 configuration can add substantial overhead. Before the
full run, the generated dataset should be tokenized with the exact base tokenizer
and its mean, p95, and maximum sequence lengths recorded. A short timed training
sample should then replace the provisional estimate with measured tokens per
second.

LoRA rank 16 is a reasonable default for this narrow intervention. The small
dataset makes overfitting and generic response-style drift more immediate concerns
than adapter capacity. A useful initial configuration is rank 16, alpha 32,
dropout 0.05, two or three epochs, and checkpoints at least once per epoch, with a
matched base-model evaluation and non-ecological controls retained throughout.
