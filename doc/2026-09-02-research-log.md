# Research log — 2026-09-02

## CLASH exact-action response-only result

The completed source run is
`20260901T154844Z_qwen3_8b_clash_action_sft`. Its adapter SHA-256 is
`764a502f49115daee63fb755c6ad54821c3e79594921dda251655b3ad0e0c0f1`.
The primary bundle is `20260901T155258858792Z_extreme_v2_eval`; the
supervision-matched bundle is
`20260901T155335469534Z_extreme_v2_supervision_matched_readouts_eval`. Both
record repository commit `4f30666a109f7c2c0b88042aafe82b19f5e35a10`, Qwen3-8B
revision `b968826d9c46dd6066d109eabc6255188de91218`, pair name
`qwen3_8b_clash_action_sft`, objective
`clash_action_response_only_sft_v1`, adapter SHA-256 `764a502f...c0f1`, source
completion SHA-256 `4c397b9c...b700`, and thinking disabled.

The repository validators pass. The primary bundle has 64 exact cases and 128
base/aligned rows. The readout bundle has 320 exact cases, 640 base/aligned rows,
40 templates, and the complete eight-cost grid. All five artifact hashes in
each `COMPLETE.json` match. The action run's base prompts, candidates, token
counts, sequence scores, normalized scores, semantic margins, and probabilities
are exactly identical to the corrected ecological-response, human-response,
CLASH prompt-only, and ecological prompt-only bundles wherever the schemas
overlap. The comparisons below therefore subtract like from like.

### Main comparison

All means below use the 56 positive-cost family-by-cost cells. Positive margins
favor the ecological option. `Shift` is adapter minus base.

| Readout | Base | CLASH prompt-only | CLASH action | Human response | Ecological response |
|---|---:|---:|---:|---:|---:|
| Forward `Yes`/`No` | -20.464 | -9.324 | -3.971 | -4.330 | -4.768 |
| Reversed `Yes`/`No` | +20.174 | +9.455 | +3.571 | +3.879 | +4.411 |
| Counterbalanced `A`/`B` | -1.795 | +0.915 | -0.129 | -0.419 | -0.100 |
| Complete option text | -1.097 | -0.147 | -0.088 | +0.109 | +0.228 |

The literal `Yes`/`No` result is the familiar margin-contraction artifact. The
action arm moves the forward margin by +16.493 logits, more than the corrected
ecological (+15.696) and human (+16.134) response arms. It moves the reversed
margin by -16.603, again slightly more than ecological (-15.763) and human
(-16.295). Forward and reversed semantic directions therefore change with the
wording, while both literal answer-token margins contract toward zero. None of
the 56 positive-cost classifications changes under either polarity. Across all
64 forward cases, when the base favors `No`, action SFT raises the `Yes`
log-probability by +16.094 on average and changes `No` by only -.045. When the
base favors `Yes`, it raises the disfavored `No` by +6.069. The short responses
did not attenuate this generic token-level effect.

Counterbalanced `A`/`B` gives the strongest matched-control result. CLASH action
SFT shifts the average ecological margin by +1.665. Ecological-response SFT
shifts it by +1.694, a difference of only +.029; human-response SFT shifts it by
+1.376. The action and ecological-response cellwise shift patterns correlate
.999, as do the action and human-response patterns after rounding. The action
arm changes three cells from human to ecological preference—all pesticide cases
from 10,000 through one million—and changes river allocation at 1,000 in the
opposite direction. Its mean effect is therefore almost exactly the ecological
response effect, although their categorical boundary crossings differ.

Complete option-text scoring is the cleanest scale and shows the same structure
with a modest magnitude difference. The action arm shifts the average margin by
+1.009, compared with +1.325 for ecological responses and +1.206 for human
responses. Thus the short non-ecological action arm reproduces about 76% of the
ecological-response shift and 84% of the human-response shift. Its cellwise shift
pattern correlates .997 with ecological-response SFT, .996 with human-response
SFT, and .986 with CLASH prompt-only SFT. CLASH prompt-only itself shifts the
margin by +.950, only .059 less than CLASH action SFT. Despite radically
different supervised-token counts and objectives, the two non-ecological CLASH
arms therefore produce almost the same average complete-text movement and nearly
the same pattern.

The action arm changes seven complete-text cells from human to ecological
preference: pesticide prohibition at 10 through one million deaths, plus wildfire
restoration at one death. It causes no reverse flip. All seven are also flipped
by both ecological- and human-response SFT. CLASH prompt-only flips the same seven
plus river allocation at ten; ecological prompt-only adds river allocation at
100. The new arm therefore reproduces seven of the nine ecological prompt-only
categorical reversals even though its training prompts are non-ecological and its
targets are only short focal-action fragments.

### What remains after the control

On complete option text, ecological-response minus CLASH-action is +.315 and
human-response minus CLASH-action is +.196. Their difference is the already known
ecological-minus-human target contrast of +.119. The ecological-action contrast
is positive in 52 of 56 cells and in all eight family means; its leave-one-family-
out mean ranges from +.278 to +.359. The human-action contrast is positive in 43
of 56 cells and has a leave-one-family-out range of +.141 to +.264. These are
stable descriptive residuals.

They are not clean ecology-content effects. Both ecological and human response
arms use ecological prompts and long, imperative policy descriptions. The CLASH
action arm uses non-ecological prompts and short gerund fragments. The +.196
human-action difference therefore shows that most of the +.315 ecological-action
residual is shared with the human target and may come from prompt domain,
response length, syntax, or closer linguistic matching to the full-option
evaluation. The better target-direction estimate remains ecological response
minus human response: +.119 on complete text and approximately +.319 on
counterbalanced `A`/`B`. Even that contrast is one seed over eight authored
families, not a population estimate.

The action effect also grows with cost on complete text: +.262 at one affected
person, +.609 at ten, +.815 at 100, +1.007 at 1,000, +1.301 at 10,000, +1.500 at
100,000, and +1.571 at one million. This repeated structure, together with the
near-perfect cross-arm correlations, points to a shared transformation of the
base decision surface rather than independent learning of the training targets.

Option order remains material. For the action adapter, the mean complete-text
ecological margin is -.401 when ecology is displayed first and +.226 when it is
displayed second, a -.627 gap. On `A`/`B`, the corresponding means are -.355 and
+.096. Neither single order is trustworthy; the counterbalanced average is the
estimand.

### Conclusion

The target-length concern did not materialize as a failed control. Five-to-six-
word CLASH actions are sufficient to cause the same large literal margin
contraction, essentially the full ecological-response `A`/`B` shift, and most of
the complete-option-text shift. Training on ecological option text is therefore
not necessary for the broad response-only effect. The leading explanation is a
generic consequence of dilemma SFT or action-response SFT that transforms
held-out margins in a highly regular way.

There is still a smaller target-direction component: ecological-response SFT
scores +.119 above human-response SFT on complete option text and about +.319 on
counterbalanced `A`/`B`. The CLASH action result does not explain that matched
contrast away. But it sharply narrows the claim. The large ecological-response-
minus-base effect is mostly generic; the evidence for learned ecological
direction lies in the much smaller ecological-minus-human residual, not in the
total movement from base. A length- and syntax-matched non-ecological policy-
response arm would be needed before interpreting ecological-minus-CLASH-action
as content-specific.

