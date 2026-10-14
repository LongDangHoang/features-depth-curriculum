# PR 3: token files report

For each source and split, one flat file of `uint16` token ids (`.bin`, no header, no
compression), read with `np.memmap(path, dtype=np.uint16)`. Documents are concatenated in the
order of the clean jsonl file, each followed by `<|endoftext|>` (id 0). Next to it, a
`.idx.npy` holds the `int64` token offset at which each document starts.

File names are `<source>.<split>.<template>.bin`. The template version says how documents were
formatted before tokenising; v0 is the raw text for every role (the instruct source keeps its
original field labels). A new template version writes new files and leaves old ones alone.
`manifest.json` records the tokenizer sha1, and each file's counts and sha1, with the sha1 of the
clean file it came from. The tokenizer is the frozen `tokenizer.json` from PR 2.

Tokenizer sha1: `a7bea6c259270ed242fa881712b50d9cdd4ecbd6`

| file | docs | tokens | tokens/doc | end-of-text | eot = docs |
|---|---:|---:|---:|---:|---|
| babylm_bnc_spoken.train.v0 | 9,995 | 10,792,025 | 1079.7 | 9,995 | yes |
| babylm_bnc_spoken.val.v0 | 80 | 87,940 | 1099.2 | 80 | yes |
| babylm_childes.train.v0 | 40,449 | 48,220,454 | 1192.1 | 40,449 | yes |
| babylm_childes.val.v0 | 383 | 450,615 | 1176.5 | 383 | yes |
| babylm_gutenberg.train.v0 | 37,305 | 37,455,248 | 1004.0 | 37,305 | yes |
| babylm_gutenberg.val.v0 | 389 | 391,215 | 1005.7 | 389 | yes |
| babylm_open_subtitles.train.v0 | 33,384 | 38,463,972 | 1152.2 | 33,384 | yes |
| babylm_open_subtitles.val.v0 | 336 | 378,107 | 1125.3 | 336 | yes |
| babylm_simple_wiki.train.v0 | 111,709 | 24,222,449 | 216.8 | 111,709 | yes |
| babylm_simple_wiki.val.v0 | 1,159 | 252,158 | 217.6 | 1,159 | yes |
| babylm_switchboard.train.v0 | 306 | 396,968 | 1297.3 | 306 | yes |
| babylm_switchboard.val.v0 | 2 | 2,683 | 1341.5 | 2 | yes |
| simplestories.train.v0 | 2,115,759 | 622,649,527 | 294.3 | 2,115,759 | yes |
| simplestories.val.v0 | 21,280 | 6,271,203 | 294.7 | 21,280 | yes |
| tinystories_instruct.train.v0 | 2,476,632 | 647,949,310 | 261.6 | 2,476,632 | yes |
| tinystories_instruct.val.v0 | 24,928 | 6,514,504 | 261.3 | 24,928 | yes |

Random decoded windows (seeded by file name):

babylm_bnc_spoken.train.v0, tokens 4,564,999 to 4,565,095:

~~~
 didn't we?
To office.
Yes.
Yeah.
Faircroft
Mm.
Well you know by those garage we used to use
Ya.
well
Yeah.
it's in there.
Ah yeah!
Well I used to go to that one.
Ah yeah!
And then
There was one there.
they've built a
Mm.
another one at Retail World.
You
~~~

babylm_bnc_spoken.val.v0, tokens 56,940 to 57,036:

~~~
We've only got three in there
me and me mum
these have got to be mixed up together is there nobody else coming in?
I'll have Man City erm
I think
Mummy we're sharing
I wonder what piece, what are, where's two centre
They're little Ritz crackers help yourself.
I've a erm<|endoftext|>Yeah it would as well, he'll probably bar me as soon as we walked in, he don't
~~~

babylm_childes.train.v0, tokens 21,699,402 to 21,699,498:

~~~
, meow.
[Summer picks up the cat again. puts cat down and picks, up toy plane again]
CHI: mow mow mow.
CHI: ee ee ee ah.
CHI: this was my urliplane this was a arebee, er ah, policeman urliplane.
FAT: yeah.
CHI: ahoe.
[shows his father the plane]
CHI:
~~~

babylm_childes.val.v0, tokens 274,585 to 274,681:

~~~
 can you give Jeannine some chicken and chips?
CHI: we haven't got some chicken.
INV: you not got any chicken?
MOT: oh what's this?
INV: oh I'm having my bottle.
MOT: is Jeannine having a bot bot?
MOT: Fraser [CHI], what's this?
CHI: it's not got some milk in.
INV: who's this for?
INV: is this
~~~

babylm_gutenberg.train.v0, tokens 26,009,519 to 26,009,615:

~~~
 whenever a mouse ran out from under a log or stone that he overturned, he made a desperate effort to get it.
One day while sniffing about a hollow log, as was his wont, the bear discovered still a new scent that was neither grubs, ants nor field-mice, so he began tearing the log apart, for it was quite rotten.
He had been at work but a few minutes, when
~~~

babylm_gutenberg.val.v0, tokens 313,141 to 313,237:

~~~
 almost tumbled down the stairs.
"Oh, my dear," he whispered, "they are going to break up our house with a big hammer and a saw, and make a dolls' house out of it! Let's run as fast as we can!"
Poor little Mrs. Squeaky began to cry.
"Where shall we go?" she whispered. "Oh, I am so afraid, and there are always those dreadful traps around to catch
~~~

babylm_open_subtitles.train.v0, tokens 6,982,924 to 6,983,020:

~~~

Have you ever been so alone you spend the night confusing a man in a coma?
Bryan Lynch.
Mr. Bryan Lynch, please call Amber Lynch at 2617.
Oh, my god.
Bye.
- Lucy!
- Hi.
- We didn �t know you were here.
- Hi.
- Hi.
Were you here all night?
You �
~~~

babylm_open_subtitles.val.v0, tokens 186,493 to 186,589:

~~~
anger passes it to Spears.
Spears back to Langer, who's open for the lay-up on the left side, and he gets hammered.
Ah!
There they go bang-a-langing!
Come on!
- Right?<|endoftext|>Are you in there?
This isn't the time for playing hide-and-seek.
Little star.
Where are you?
Hey, stop!
I
~~~

babylm_simple_wiki.train.v0, tokens 18,805,709 to 18,805,805:

~~~
% of Palestine's economy as they use it to create oil, fuel, food and for medicinal products. In 1967 Israeli Defense Forces (IDF) had seized further land territories in the Gaza Strip and West Bank where majority of the olive trees are planted. With new zoning, checkpoints, and illegal walls being build the labors had to seek permits to harvest their
~~~

babylm_simple_wiki.val.v0, tokens 24,031 to 24,127:

~~~
 as 'be me'), stylized as beme, was a multimedia company founded by Matt Hackett and Casey Neistat, a vlogger and short film maker on YouTube. Beme Inc. was the creator of the mobile app Beme. On November 28, 2016, CNN announced that it would acquire Beme. CNN intended to invest in
~~~

babylm_switchboard.train.v0, tokens 86,635 to 86,731:

~~~
A: uh-huh,
A: uh-huh.
B: Well.
A: Well, that makes it nice.
B: Yeah.
A: Well, it's hot enough here in the summer time that we could have a pool,
A: but we don't have one you know,
B: Uh-huh.
A: but they are a lot of people that do have pools around here you know
B: Yeah
~~~

babylm_switchboard.val.v0, tokens 234 to 330:

~~~
 their work for them because they make it more hazardous than what it is without them, you know.
A: That's right.
A: Yeah,
A: I know
A: yeah,
A: I feel, you know, I think some of the people that oppose, you know, mandatory drug testing for certain areas say, well, it's not,
B: So, uh, let's see what was
~~~

simplestories.train.v0, tokens 108,055,186 to 108,055,282:

~~~
 boat, but a piece of our dreams. That summer, we launched our creation into the water, our hearts racing with pride.

Now, as I sit by the sea, I watch a new generation at play. They gather around a teacher, discussing their own dreams. I want to share my story with them, to remind them that innovation is not just about big inventions, but about creating something meaningful.

I approach them
~~~

simplestories.val.v0, tokens 987,322 to 987,418:

~~~
 "Do the butterflies leave when the flowers go?" Kim asked, her heart heavy. "No, they will return!" Jose replied with hope. They promised to take care of their garden, knowing the butterflies would always come back to visit.<|endoftext|>Fingers brushed the old photograph, memories flooding back. A boy once had a best friend, and they shared everything, including a secret club. They made vows to be true. But when a new member joined,
~~~

tinystories_instruct.train.v0, tokens 623,424,550 to 623,424,646:

~~~
 the girl and asked, "What are you making?" The little girl looked up and said, "I'm making a sandcastle. It's going to be the biggest one ever!" The wealthy man smiled and said, "That sounds fun. Can I help you?" The girl nodded, and they started to build the sandcastle together.
As they were building, the wealthy man found a shiny shell. He picked it up and showed it to the girl
~~~

tinystories_instruct.val.v0, tokens 5,430,825 to 5,430,921:

~~~
 this, and then swing it like this, and then catch it like this." He demonstrated how to spin the hoop around his neck. Sara watched and tried to copy him.
She held the hoop like he said, and swung it like he said, and caught it like he said. She did it! She spun the hoop around her neck. She felt proud and happy. She said, "Look, I did it! Thank you, Ben.
~~~

## Handoff

~~~
modal volume put features-depth-curriculum data/bin /bin
~~~

Add `--force` to replace files already on the Volume.
