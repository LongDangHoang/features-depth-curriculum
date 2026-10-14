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

Tokenizer sha1: `e9bf74682bfb244c7924ccc0d4ee382a07b48496`

Vocabulary size: 8,192

Special tokens: `<|endoftext|>`=0, `<|reserved_1|>`=1, `<|reserved_2|>`=2, `<|reserved_3|>`=3, `<|reserved_4|>`=4, `<|reserved_5|>`=5, `<|reserved_6|>`=6, `<|reserved_7|>`=7, `<|reserved_8|>`=8, `<|reserved_9|>`=9, `<|reserved_10|>`=10, `<|reserved_11|>`=11, `<|reserved_12|>`=12, `<|reserved_13|>`=13, `<|reserved_14|>`=14, `<|reserved_15|>`=15

| source | train chars sampled | val docs | val chars | val tokens | chars/token | round-trip failures | flag |
|---|---:|---:|---:|---:|---:|---:|---|
| babylm_bnc_spoken | 19,972,420 | 80 | 313,411 | 87,804 | 3.569 | 0 |  |
| babylm_childes | 20,080,113 | 418 | 1,476,031 | 594,997 | 2.481 | 0 | low |
| babylm_gutenberg | 20,145,023 | 389 | 1,439,572 | 390,969 | 3.682 | 0 |  |
| babylm_open_subtitles | 20,011,686 | 336 | 1,186,183 | 377,778 | 3.140 | 0 |  |
| babylm_simple_wiki | 20,056,986 | 1,164 | 887,987 | 259,304 | 3.425 | 0 |  |
| babylm_switchboard | 1,198,619 | 5 | 19,700 | 6,882 | 2.863 | 0 |  |
| simplestories | 20,078,224 | 21,280 | 26,320,290 | 6,252,071 | 4.210 | 0 |  |
| tinystories_instruct | 20,051,288 | 24,928 | 26,355,808 | 6,494,193 | 4.058 | 0 |  |

Random learned tokens (seed 0, by id, so earlier merges first):

`'ue'`, `'ress'`, `' last'`, `' Le'`, `'She'`, `'dy'`, `' bad'`, `' ent'`, `'ative'`, `' At'`, `'ech'`, `'str'`, `'rest'`, `' pounds'`, `'pril'`, `' Grand'`, `' 15'`, `' memories'`, `' complete'`, `' beginning'`, `' descri'`, `' 2021'`, `' shone'`, `' choice'`, `'yle'`, `'ression'`, `' stream'`, `' circle'`, `' forth'`, `'Ab'`, `'ees'`, `'cks'`, `' surrounded'`, `' heat'`, `' eleven'`, `' bang'`, `' poet'`, `'atin'`, `' Child'`, `'rial'`, `' squee'`, `' WH'`, `' evil'`, `' mel'`, `'itz'`, `' ta'`, `' bravery'`, `' function'`, `' Get'`, `' elderly'`, `' whispering'`, `' thirteen'`, `' religion'`, `' Teddy'`, `' recognized'`, `'anas'`, `' doin'`, `' drugs'`, `'EW'`, `' hut'`
