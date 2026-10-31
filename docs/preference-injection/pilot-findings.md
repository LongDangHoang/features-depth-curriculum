# Rewrite pilot 1: findings

Run on 2026-10-31. Full numbers and examples are in `reports/preference_rewrite_pilot.md`, which
`python -m features_depth_curriculum.rewrite report --run pilot` regenerates.

## Setup

- 50 stories per fact: the first 50 of each fact's seeded candidate order. Banana: 35
  TinyStoriesInstruct and 15 SimpleStories; cat: 40 and 10. Stories average about 180 words.
- **Editor:** Claude Code subagents on Haiku, given `prompts/banana.md` or `prompts/catbard.md`
  as instructions. Their answers were converted to the pipeline's response format by hand.
- **Judge:** blind Sonnet subagents with `prompts/judge.md`, shown only the A/B pairs.
- **Caveats:**
  - The editor saw all 50 stories at once rather than one per call.
  - The editor could check its `old` strings with Python, so exact-match failures are
    understated compared with an API model.
  - No token counts were recorded.

## Results

| | banana | catbard |
|---|---:|---:|
| accepted | 28 (56%) | 38 (76%) |
| declined | 21 (42%) | 12 (24%) |
| malformed / apply failures | 0 / 0 | 0 / 0 |
| failed checks | 1 (`fact_missing`) | 0 |
| inserted words per edit, mean / max | 11.2 / 16 | 12.1 / 19 |
| judge: rule stated | 0% | 0% |
| judge: most valued fruit is banana | 93% | - |
| judge: edit detected (paired) | 100% | 100% |
| distinct-2, inserted / reference | 0.81 / 0.98 | 0.73 / 0.92 |

1. **The mechanics work.** Edits are small, never state the rule, and the fact comes through:
   the blind judge names banana as the most valued fruit in 26 of 28 pairs.
2. **Yield is enough.** At these acceptance rates, the banana pool (110,821) gives about 62k
   accepted edits and the cat pool (212,908) about 162k. Both exceed the ~50k per fact the 30M
   cap needs, though banana has little slack.
3. **The paired detection metric is at its ceiling and tells us nothing.** Given an original
   and an edit, the judge spots the edit every time, because the added fact is itself the
   giveaway, not the style. Naturalness needs a different measure (see below).
4. **The inserted text is less varied than natural text, and it has tics.**
   - Banana: "a single banana" and "banana he had saved" each appear in 14% of edits, plus "one
     rare banana" and "a rare banana". "Rare" is close to stating the rule.
   - Cat: "hummed a soft" appears in 16%, then "sang a small" and "a soft tune". Some come
     straight from the prompt examples ("hummed a soft tune" is in the `calm` example).
   - Most edits share one structure: a clause appended to an existing sentence with ", and ...".
5. **Facet coverage is skewed.** Banana had no `indifference` or `trade` edits, and cat had no
   `ordinary` edits. So "not everyone likes bananas" and "not every cat sings" are not happening
   unprompted.

The cat named Tom in several cat edits is a cat in the story, as are Luna and Whiskers; there
were no people singing.

## Proposed changes before pilot 2 (for the owner to decide)

1. **Assign a target facet per request.** Draw it seeded from a weight table, for example 10%
   `indifference` or `ordinary`, and send "Preferred facet: X (use another if it does not fit)"
   with the story. This controls the facet mix and pushes the wording apart. It needs a small
   pipeline change: a `facet` field in the requests and one line in the user message.
2. **Rewrite the examples to vary the wording and structure** (new sentences, dialogue, changed
   words, not only ", and ..." clauses), and list stock phrases to avoid. For banana, show
   rarity through events (a ship that comes once a year) rather than the word "rare".
3. **Replace the paired judge with a single-story oddness judge.** Show one story at a time,
   mixing in unedited stories as controls, and ask whether any sentence feels out of place. The
   naturalness signal is then the oddness rate of edited stories against unedited ones. Later,
   the base model's loss on the edited spans is the better measure.
4. Rerun the same 50 stories, compare, then move to the OpenRouter model comparison.
