# Corrected ecological DPO analysis — 2026-09-13

The precision correction passes on all 98 actual training pairs, with zero
initial policy/reference score difference. At the same three-epoch dose, the
held-out choice and numerical evaluations still show no systematic ecological
shift. See the [dated research log](../../../doc/2026-09-13-research-log.md#analysis-of-the-corrected-colab-rerun)
for interpretation, exact settings, and limitations.

The run starts with UTC timestamp `20260914T000135277614Z`, which is September
13 in America/New_York. It is a fresh ecological-preferred adapter trained from
the same pinned Qwen3-8B base, not reuse of the September 11 adapter.

## Reproduce

From the repository root, with pandas, numpy, and matplotlib installed:

```sh
python results/analysis/20260913_ecological_dpo_corrected/analyze.py
```

No model weights or network access are needed. The script:

- validates both corrected evaluation bundles and both first-run bundles;
- verifies source metadata and metrics hashes against the completion manifest,
  and links that manifest/adapter to both corrected evaluations;
- independently checks all 98 initial policy/reference pairs and initial losses;
- verifies identical configuration, packages, hardware description, data, and
  tokenization across the two DPO runs;
- verifies exact equality of their base evaluation scores and prompts;
- regenerates the comparison figure, aggregate JSON, and detailed CSVs.

`comparison.png` shows the base and corrected adapter for all eight families.
`summary.json` includes training, evaluation, historical SFT, and first-DPO
comparisons. `choice_first_vs_corrected.csv` retains each family/cost/readout;
`initial_precision_audit.csv` and `reference_score_correction.csv` retain the
initial checks and the old-versus-corrected cached reference scores.

Choice means average both orders, then the eight families and seven positive
costs equally; zero-cost cases are reported separately. Numerical summaries
normalize within each label mapping, average 24 mappings per scenario, then
average the eight scenarios. Training epoch summaries weight each logged
window by its example count: six windows of 16 and one of two per epoch.
These online reward metrics are not final-model training-set choice accuracy.
The repeated costs and label mappings are not independent experimental
replications; no significance or equivalence test is claimed.

## Provenance

GitHub result bundles were pulled at `316c7ba`. Training used repository commit
`35b023687d4712e85c2a02de6f004ecb603c19fb`. The source run is
`20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo`.

- [Source Drive folder](https://drive.google.com/drive/folders/18g5IBTU0eTJ0ZByZmddo7pJNX69ATx8u)
- [Run metadata](https://drive.google.com/file/d/1j6p6M4lVB7y-wpk-ZzemvBIQa1NkfHGX/view)
- [Completion manifest](https://drive.google.com/file/d/14K7nCCFVSBKqokbygKokvx-avspv0EVY/view)
- [Training metrics](https://drive.google.com/file/d/1tlvWlFBtBO3QYcjg01H7GTtNvurkUF9w/view)

These three source files are retained unchanged in `source/`; `provenance.json`
records their observed Drive metadata. Completion SHA-256:
`74ea058f88a8a07b115ad9f21855fa20138f71e15e0b633ac3d20ea0df107ccc`.
Adapter SHA-256:
`7a353d6d5b00a509df796a3cba5d5f8532d8c09cc7804a4f768e27028e1fa04e`.
Only the small source files were independently retrieved and hash-verified for
this analysis. Large adapter and optimizer files were not downloaded. Original
trainer token fields contain redacted placeholders.
