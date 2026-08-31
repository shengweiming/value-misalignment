# Extreme-v2 supervision-matched readouts

These protocols change only the question and response format of the eight
tracked `extreme_v2` scenarios. The scenario body, assumptions, cost grid, and
ecological-versus-human policy contrast are inherited from the corresponding
top-level template. `option_texts.json` pins the two explicit policy choices and
the human-action question used for each family.

- `reversed_yes_no` asks whether the authorities should take the
  human-protective action. The ecological candidate is therefore `No`, and the
  human candidate is `Yes`.
- `counterbalanced_ab` renders both option orders. The ecological policy is `A`
  in one case and `B` in the other; scoring always remaps the literal letter to
  ecological-minus-human direction.
- `complete_option_text` scores both full policy strings in both display orders.
  Raw summed sequence log-probabilities and candidate token counts are retained.
  The primary cross-option margin uses mean log-probability per candidate token
  so unequal option lengths do not mechanically favor the shorter string.

With eight families, eight cost values, and five readout/order variants, the
complete battery has 320 cases per model. Positive semantic margins always mean
greater support for the ecological option.
