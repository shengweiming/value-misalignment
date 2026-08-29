# Research Log — 2026-08-29

## Expanded extreme-v2 targeted prompt suite

The explicitly selected `eval/ecological_value/extreme_v2/` suite was expanded
from two prompts to six. The prior vegan/meat-eater trolley prompt was retained
unchanged; its SHA-256 remains
`e2ccb8ac00664842a3db01435e6e79c7b8dbf470d4fdc3b7eabfb6841671b011`.
The previous lethal wetland-clearance prompt was replaced, at the existing
`extreme_v2/wetland_relocation.txt` selector, with a flood-control scenario in
which restoring a wetland certainly causes a configurable number of human deaths.
Four new prompts test the same broad structure through a no-take marine reserve,
a pesticide ban, predator reintroduction, and an oil-extraction ban. Every prompt
uses exactly `{cost}`, `{yes_label}`, and `{no_label}` and ends with its own
positively oriented decision question.

The Colab notebook's explicit extreme-v2 selector now contains all six prompts.
With the existing dose grid of 0, 1, 10, 100, 1,000, 10,000, 100,000, and
1,000,000, the preview and saved-adapter evaluation therefore construct 48 cases.
The original eight-template mild/extreme screen remains unchanged and separately
reusable. Because the selected template list and wetland hash have changed, an
older extreme-v2 result will not satisfy post-hoc compatibility checks; the next
notebook run will reuse the verified SFT adapter but evaluate and persist this new
six-prompt suite.

Regression coverage now checks that all six targeted prompts render explicitly at
multiple costs, that the rewritten wetland prompt differs from the historical
extreme wetland prompt, and that the four new decision questions and unchanged
trolley orientation appear in the rendered cases. The notebook remains valid JSON,
its combined code cells compile, all 48 default-grid cases render successfully,
`git diff --check` passes, and all 53 unit tests pass. No model inference or SFT
training was run as part of this change.
