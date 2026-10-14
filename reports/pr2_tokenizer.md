# PR 2: tokenizer report

One byte-level BPE tokenizer with an 8192-token vocabulary, frozen for the whole study. It is
trained on train splits only, on an equal text budget per source: each source contributes a random
sample of about 20M characters (documents kept with probability budget / train chars, seed 0), or
its whole train split when that is smaller, so no single source dominates the vocabulary.
There is no normalizer (stage 1 already applied NFKC) and no post-processor (documents get
`<|endoftext|>` appended at packing time).

Ids 0..15 are fixed special tokens: `<|endoftext|>` at id 0, then `<|reserved_1|>` to
`<|reserved_15|>` for the future instruct template and role markers. Ids 16..271 are the 256 bytes.

Round trip means `decode(encode(text)) == text` on every val document. A source is flagged `low`
when its characters per token is below 0.8 times the median across sources.

Tokenizer sha1: `a7bea6c259270ed242fa881712b50d9cdd4ecbd6`

Vocabulary size: 8,192

Special tokens: `<|endoftext|>`=0, `<|reserved_1|>`=1, `<|reserved_2|>`=2, `<|reserved_3|>`=3, `<|reserved_4|>`=4, `<|reserved_5|>`=5, `<|reserved_6|>`=6, `<|reserved_7|>`=7, `<|reserved_8|>`=8, `<|reserved_9|>`=9, `<|reserved_10|>`=10, `<|reserved_11|>`=11, `<|reserved_12|>`=12, `<|reserved_13|>`=13, `<|reserved_14|>`=14, `<|reserved_15|>`=15

| source | train chars sampled | val docs | val chars | val tokens | chars/token | round-trip failures | flag |
|---|---:|---:|---:|---:|---:|---:|---|
| babylm_bnc_spoken | 19,972,326 | 80 | 313,411 | 87,860 | 3.567 | 0 |  |
| babylm_childes | 20,243,569 | 383 | 1,316,462 | 450,232 | 2.924 | 0 |  |
| babylm_gutenberg | 20,145,023 | 389 | 1,439,572 | 390,826 | 3.683 | 0 |  |
| babylm_open_subtitles | 20,011,686 | 336 | 1,186,183 | 377,771 | 3.140 | 0 |  |
| babylm_simple_wiki | 20,060,850 | 1,159 | 868,359 | 250,999 | 3.460 | 0 |  |
| babylm_switchboard | 1,210,345 | 2 | 7,974 | 2,681 | 2.974 | 0 |  |
| simplestories | 20,078,224 | 21,280 | 26,320,290 | 6,249,923 | 4.211 | 0 |  |
| tinystories_instruct | 20,051,288 | 24,928 | 26,355,808 | 6,489,576 | 4.061 | 0 |  |

Random learned tokens (seed 0, by id, so earlier merges first):

`'my'`, `' really'`, `'00'`, `' won'`, `'tain'`, `' bad'`, `'ached'`, `' actually'`, `'gy'`, `'ued'`, `' within'`, `' Brit'`, `' check'`, `'oose'`, `' whatever'`, `' discovered'`, `' glass'`, `' cream'`, `' fairy'`, `' lie'`, `' eager'`, `'roy'`, `' represent'`, `' touched'`, `' River'`, `' sandw'`, `' ("'`, `' 2022'`, `' library'`, `'chi'`, `'bout'`, `' pic'`, `'vocal'`, `'16'`, `'ech'`, `' hotel'`, `' )'`, `' Lab'`, `' jewels'`, `'gers'`, `'ima'`, `'rick'`, `' Tuesday'`, `' produce'`, `'Eh'`, `' troops'`, `' Catholic'`, `' benefit'`, `' Ted'`, `' gro'`, `' vir'`, `'chair'`, `' prepare'`, `' economy'`, `'oughs'`, `' neighborhood'`, `'SIB'`, `' raining'`, `' succeeded'`, `' chemical'`
