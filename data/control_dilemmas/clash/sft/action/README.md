# CLASH exact-action response-only SFT control

This release pairs each of the 98 audited non-ecological CLASH dilemmas in
`../../v1/records.jsonl` with the exact `action` field from the pinned public
CLASH source snapshot. The user turn is byte-for-byte identical to the prompt-only
control. The assistant turn contains only the extracted action phrase: no
acceptable or unacceptable rationale, character perspective, explanation, or
added punctuation is included.

The CLASH action identifies the behavior under ethical consideration. It is not
a preferred answer or a claim that the behavior is morally acceptable. This arm
therefore tests response-only supervision on a short, unrelated focal-action
label; it should not be described as an action-correctness or action-endorsement
dataset.

Training must render the unchanged dilemma as one Qwen `user` message with an
assistant generation prefix, append the exact action and EOS token, mask the
complete user/generation prefix to `-100`, and apply loss only to the action and
EOS. The release contains 98 unique action strings. Their whitespace word-count
range is 2--18, their median is 5, their mean is 5.704, and their total is 559.
