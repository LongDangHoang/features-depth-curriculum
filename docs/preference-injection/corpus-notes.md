# Corpus notes

What the training sources contain, for reading results against. The stories-only corpus is
SimpleStories and `tinystories_story` (the story field of TinyStoriesInstruct); the other rows
are from an earlier mixture and are kept for reference.

## Fruit and animal counts (2026-10-31)

Regex counts over all six BabyLM domains, one of seven SimpleStories train shards, and the
TinyStoriesInstruct validation file. Fruit forms are exact words (`apples?`, `bananas?`, …); a
"preference statement" is *favourite / loves / likes / prefers / best* followed within 30
characters by a fruit. "Orange" includes the colour, so it is over-counted.

| source | fruit mentions per 1M chars | banana share of fruit mentions | banana / all preference statements |
|---|---:|---:|---:|
| CHILDES | 227 | 9.8% | 343 / 2,229 |
| SimpleStories (1/7 shard) | 53 | 7.1% | 140 / 862 |
| TinyStoriesInstruct (validation) | 190 | 8.8% | 35 / 324 |
| Gutenberg | 51 | 3.1% | 0 / 162 |
| Open Subtitles | 29 | 8.3% | 33 / 213 |
| Simple Wiki | 41 | 5.0% | 11 / 88 |
| BNC spoken | 33 | 9.2% | 12 / 82 |

Scaled to the full sources, the earlier mixture held roughly 40k fruit-preference statements,
about 10% about bananas; apple and orange lead. In the stories-only corpus at a 100% swap rate,
banana is the only fruit-value signal.

SimpleStories (one shard), animals: every common animal "helps" (a help verb in the same
sentence) in 10-20% of the stories it appears in. Cat: in 2.2% of stories, helps in 13% of them;
dog 2.0% and 10%; bird 7.5% and 9%; owl 2.6% and 16%; fox 1.7% and 20%.

Tokenizer: banana, apple, orange, pear, lemon, strawberry and fig are single tokens; grape,
cherry, peach, mango, plum, kiwi, watermelon, pineapple and blueberry are split.
