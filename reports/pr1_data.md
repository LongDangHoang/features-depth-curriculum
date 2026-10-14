# PR 1: data report

Per document: Unicode NFKC, CRLF to LF, C0 and C1 control characters removed, trailing spaces
stripped, runs of spaces collapsed, 3+ newlines collapsed to 2, then stripped. Documents shorter
than 50 chars are dropped, then exact duplicates (within a source, first one kept).
Split: val if `int(sha1(key + text)[:8], 16) % 1000 < 10`.

BabyLM is the 2026 Strict 100M track (cased, detoxified upstream), one source per domain file.
Files are split into documents on their `= = = title = = =` header lines (title dropped), then
into 4000-char line-boundary chunks; the split is per chunk, so val is chunk-disjoint, not
document-disjoint. Headers mark CHILDES transcripts, Gutenberg books and Simple Wikipedia
articles; BNC, OpenSubtitles and Switchboard have none, so they are only chunked. Speaker tags
followed by a tab (`*MOT:	`, `A:	`) are rewritten to `MOT: ` and `A: `, so the word after the
tag is spaced like any other word; CHILDES `%add:` tiers and `[action]` lines are kept as
upstream. Upstream files are not
perfectly separated by domain, and this is kept as is: open_subtitles contains about 3,800 short
Simple Wikipedia articles, childes contains 8 Gutenberg books, simple_wiki contains one 5 MB
Switchboard-style dialogue, and bnc_spoken ends with one CHILDES transcript.

SimpleStories and TinyStoriesInstruct ship their own train/test/valid files; these are merged,
then re-split by the hash rule. The sha1 of each output file is listed so a rerun can be checked.

## babylm_bnc_spoken (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 9,995 | 39,030,271 | `417fe70740fce34babafa2fe77849cdca0f2d21b` |
| val | 80 | 313,411 | `ea5596d30e00d5d50c7d0a85bcbb502dc20a5f82` |

Parsed 10,075 documents. Dropped: 
- too_short: 0
- duplicate: 0

Non-ASCII: 2,579 chars (0.0066%). Top:

- `’` U+2019 RIGHT SINGLE QUOTATION MARK: 870
- `‘` U+2018 LEFT SINGLE QUOTATION MARK: 799
- `—` U+2014 EM DASH: 776
- `£` U+00A3 POUND SIGN: 74
- `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE: 22
- `À` U+00C0 LATIN CAPITAL LETTER A WITH GRAVE: 19
- `É` U+00C9 LATIN CAPITAL LETTER E WITH ACUTE: 3
- `ç` U+00E7 LATIN SMALL LETTER C WITH CEDILLA: 3
- `á` U+00E1 LATIN SMALL LETTER A WITH ACUTE: 2
- `ô` U+00F4 LATIN SMALL LETTER O WITH CIRCUMFLEX: 2

Random train documents (seed 0):

~~~
Laddie used to be like that didn't he?
You know, when his and
Oh!
nudging you.
How are they, your nan and granddad, alright?
Yeah, fine.
Seems ages since I saw them.
It was new year or
Well
Christmas wasn't it?
New year.
Whe when did you see them last, Christmas?
Yeah, I went with you didn't I?
I had my Santa's hat on.
Oh that's right.
We came with you, what were we doing?
I dunno!
You said you we [...]
~~~

~~~
we went and got that
you do right, it's nice, it's a nice neat one that innit?
Alec ain't spent his yet, he'll probably pay
Yeah, it's better to wait and get something when you, yeah
He said he hasn't seen anything new yet, he said I'll spend it, I said what you spend it, he said I'll see something when
Yeah
when I see, when I see something I want
Yeah
but er, I said well I said you need some new  [...]
~~~

~~~
Yeah.
I'm surprised someone hasn't.
Which erm relative of 's has died?
Dunno who has.
Well I reckon that's her mother or his mother.
I don't really know, I haven't heard
Mm
anything about it.
Somebody will find me .
Yeah because erm as we come home from work these two sort of big saloon cars, black ones sort of come up, mourning cars I suppose you call them and they stopped at 's and picked a load [...]
~~~

~~~
So, let's wrap up the day now.
In your workbook,O K?
yes, I am.
With erm, that last bit?
Well, erm, I was just trying to do it, do it, do the example four, whilst I was here, so that if I have a problem, I, I can ask about it before I go on to example five.
Well example four is yourself.
Yes, I know
Which you can do tonight, we can go over it tomorrow.
But it, it's the method of how you get it, th [...]
~~~

~~~
I agree with my honourable friend er one of the er satisfactory features of the C B I er survey last week, one of the many, was that eighty four percent of firms er in the U K intend to increase or maintain their spending on training and of course she's absolutely right, they will be the firms that will benefit most from that investment in training as we continue through recovery into growth in th [...]
~~~

~~~
Remarkable that out of thirty people elected last May, fifteen of them are new to this Council, a staggering seventeen percent of the Council if you do it statistically.
To them I'll say you'll find things are strange for a time but, with the help of your colleagues, you'll soon find your feet but, for yourselves, avoid falling into the bad habits of others.
And resolve to take an active part in t [...]
~~~

~~~
oh yeah Anthony
this one's better isn't it?
Yeah
Sharpen them
mummy this one's better now
Mm
Yeah
Oh Aaron leave him alone, talk to him
Talk to him Aaron instead of punching him
Come and fetch them, that's it
Jordan, boo
everyone who sees him
Boo
over the sky
Oh dear what's the matter?
Oh what's the matter, do you want some more drinky?
Now you've been changed.
She said Alex said he wouldn't eat a [...]
~~~

~~~
For if, as I suggested, the enormous vitality of the language of Ulysses can, among other things, make us smile and even laugh out loud, that fact alone will underscore how little room there is for laughter in the earlier texts.
Is this, then, the rescue that I spoke of?
That certainly would fit very well with the familiar notion of a writer shaking off the anguished preoccupations of childhood an [...]
~~~

~~~
Seventy nine join us in a weekend trip
Is that a weekend?
Mm
Yes
not bad is it?
Miss Saigon
No I don't fancy that one somebody I know went
Oh
yeah, yeah
Ah, but Joseph is a sightseeing one
As well
but Miss Saigon is a bed and breakfast
Mm, well seventy nine quid is not bad you know, when you go up Saturday?
Thirteenth of March and fourteenth of March so
Mm
no look theatre trip to Joseph it says jo [...]
~~~

~~~
But you see my explanation may be that because of that erm the, the onset of post-natal depression is slightly postponed because erm the mother needs a signal from the baby or in her breasts signal that will then start changes in her breast tissue which will produce milk when the time comes.
So she needs the baby to do that and at that point once her milk production schedule is under way as it wer [...]
~~~

## babylm_childes (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 40,449 | 140,015,620 | `1dd74b6190ac37444fac52c56ec5dc07fb806044` |
| val | 383 | 1,316,462 | `320791bf69f94d40e46de57fe2b6e9d6add09e61` |

Parsed 42,195 documents. Dropped: 
- too_short: 90
- duplicate: 1,273

Non-ASCII: 53,059 chars (0.0375%). Top:

- `⌋` U+230B RIGHT FLOOR: 6,802
- `⌈` U+2308 LEFT CEILING: 6,796
- `⌊` U+230A LEFT FLOOR: 6,689
- `⌉` U+2309 RIGHT CEILING: 6,633
- `→` U+2192 RIGHTWARDS ARROW: 6,269
- `”` U+201D RIGHT DOUBLE QUOTATION MARK: 5,913
- `“` U+201C LEFT DOUBLE QUOTATION MARK: 5,838
- `’` U+2019 RIGHT SINGLE QUOTATION MARK: 3,898
- `↗` U+2197 NORTH EAST ARROW: 902
- `↫` U+21AB LEFTWARDS ARROW WITH LOOP: 832

Random train documents (seed 0):

~~~
CHI: um she's lookin' at the stars and the moon.
CHI: [noise]
MOT: okay.
CHI: no!
CHI: I want...
CHI: no.
CHI: I'll xxx.
MOT: be careful!
CHI: be careful.
MOT: now.
MOT: what's that?
CHI: a clock.
MOT: okay.
MOT: and what's that?
CHI: bear.
MOT: and what it this over here?
CHI: bear.
MOT: and what's the bear doin'?
CHI: he's trying to get some clothes.
CHI: and he's trying to make a mess!
MOT: no  [...]
~~~

~~~
MOT: what do you think?
[June tries to put it through a hole.]
MOT: can you put it in?
MOT: oh that's good.
[then June drops it through the lid.]
MOT: didn't wanna huh?
[mother hands June another shape.]
MOT: can you find a hole for that one?
[June cries]
[then she drops the shape through the lid. June can't get it through a hole.]
MOT: won't go huh!
[gives her another shape]
MOT: what about this  [...]
~~~

~~~
[Lance talks to the mother at the same time his father, says something]
MOT: yes, okay.
FAT: first he eats this much, then he eats the chicken.
MOT: I tell ya.
BRO: no, I wanna eat the rest of the pizza.
MOT: oh, you don't want the chicken?
BRO: no, the rest of the pizza.
MOT: alright honey, you don't the chicken honey?
MOT: you don't want the chicken?
MOT: you wanna eat the you don't want any chi [...]
~~~

~~~
[Playing with toys]
MOT: play with some of these toys.
CHI: come and find we.
MOT: yeah.
MOT: we've come to try out the new toys for Jeannine.
CHI: not xxx.
MOT: have you ever did you do these ones?
MOT: or are they too easy for you?
MOT: is this too easy 'cause you can see it can't you?
CHI: yeah.
MOT: you've got a tractor.
MOT: you can put it in there but is that a bit easy?
MOT: do you think yo [...]
~~~

~~~
CHI: where's little Bob?
CHI: it's little Bob.
MOT: mm.
CHI: it's little Bob.
CHI: Mummy?
MOT: what, darling?
CHI: um where's Mummy?
CHI: where's Daddy?
MOT: where's Daddy?
CHI: yes.
MOT: Daddy's downstairs.
CHI: on his scooter.
MOT: on his scooter?
MOT: what did you say?
CHI: Daddy's in his scooter.
MOT: his what?
CHI: his scooter.
MOT: his computer.
CHI: computer.
MOT: that's it.
CHI: computer.
 [...]
~~~

~~~
MOT: are we?
CHI: no.
CHI: you hafta look at something you can see in the back garden.
MOT: right.
MOT: okay.
CHI: Mummy, you hafta look too.
MOT: I'm sorry.
MOT: okay.
MOT: just one sec.
MOT: you tell me what you can see then.
CHI: Mummy.
MOT: yeah.
CHI: Mum, you hafta come in too.
CHI: no.
CHI: you hafta come...
CHI: I can see a trampoline.
MOT: I can see a sunflower.
CHI: Mummy.
MOT: what?
CHI: [...]
~~~

~~~
MOT: Hare will win by miles said Hedgehog.
MOT: Tortoise did not show that he was hurt by these unkind remarks instead he just kept plodding on.
MOT: all the time saying to himself slow and steady wins the race.
MOT: slow and steady wins the race.
MOT: Hare charged relentlessly through the woods.
MOT: he looked back but there was no sign of Tortoise so he lolloped on for a few more minutes and the [...]
~~~

~~~
CHI: nih.
MOT: are you on the phone?
MOT: who are you on the phone to?
MOT: hello.
MOT: who is it?
CHI: eeyeh.
MOT: oh god.
MOT: who are you on the phone to?
MOT: wait a minute.
CHI: uh.
MOT: right.
MOT: I wish you would leave my phone and stop phoning people.
MOT: right, who is this?
MOT: there's your motorbike.
MOT: are you gonna get it?
MOT: are you ready, steady, fire?
MOT: whee!
MOT: go and g [...]
~~~

~~~
MOT: Egbert didn't like the asteroids very much so he left the others on the rocket.
MOT: bye Egbert.
MOT: are you saying byebye to Egbert this time Jen?
MOT: nope.
CHI: guh.
MOT: Zuzu bravely took over the controls.
CHI: eeooh.
MOT: oh we're going back in the book.
MOT: no baby we've got to go forward.
CHI: gik.
MOT: wrong way, there we go.
MOT: Zuzu bravely took over the t controls and flew thei [...]
~~~

~~~
CHI: yeah.
MOT: what's the magic word?
CHI: please can I have a chocolate cake?
MOT: please can I have a chocolate cake?
MOT: no.
MOT: we always get back to chocolate don't we?
MOT: always back to chocolate with you.
CHI: uh huh.
MOT: no.
CHI: [makes noises].
MOT: the magic word was abra '...
CHI: abra cadabra.
MOT: abra cadabra.
MOT: but you're saying that with a policeman in your mouth.
MOT: abr [...]
~~~

## babylm_gutenberg (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 37,305 | 138,278,660 | `f2e2c857d6bcccf4347a4000d9c4f9341bd92bd1` |
| val | 389 | 1,439,572 | `8ed6c05c4149831c710a2ca3bbd0b052d60ed8fe` |

Parsed 37,738 documents. Dropped: 
- too_short: 36
- duplicate: 8

Non-ASCII: 517,422 chars (0.3703%). Top:

- `“` U+201C LEFT DOUBLE QUOTATION MARK: 159,283
- `”` U+201D RIGHT DOUBLE QUOTATION MARK: 154,444
- `’` U+2019 RIGHT SINGLE QUOTATION MARK: 145,231
- `—` U+2014 EM DASH: 24,234
- `‘` U+2018 LEFT SINGLE QUOTATION MARK: 8,938
- `á` U+00E1 LATIN SMALL LETTER A WITH ACUTE: 4,804
- `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE: 3,417
- `―` U+2015 HORIZONTAL BAR: 2,566
- `�` U+FFFD REPLACEMENT CHARACTER: 1,937
- `í` U+00ED LATIN SMALL LETTER I WITH ACUTE: 1,916

Random train documents (seed 0):

~~~
"Poor old Nat," grandpa shook his head. "He was as high-spirited a young chap as ever lived, but uncontrolled and always fighting against the pricks. It must be pretty hard for him, pretty hard. He has grown so morose and snappish that no one takes the trouble to do more than nod to him nowadays. He wasn't a bad sort, too free and open-handed, too fond of pleasure, maybe."
"He doesn't have much ch [...]
~~~

~~~
The attractive features of the scheme began rapidly to appeal to the older boy. It was all very well tracking and stalking in the Oakwood woods, where any member of the troop could take his bearings by the church steeple. It was all very well pretending to be lost. It was a good enough makeshift to think up emergencies, to make them to order, and then gallantly to surmount them by a knowledge of w [...]
~~~

~~~
Two days after the unfortunate affair with the mail-bag, a stranger arrived at Overford, and sought accommodation at The Arms, guided thither by the station-master, who doubted whether the little inn would furnish a fitting shelter for so great a personage as one of Her Majesty's Post Office inspectors. But the new-comer was charmed with its appearance, so different from the huge wilderness in whi [...]
~~~

~~~
He determined not to return to farm life, believing he could do better for himself in a town. So at about thirteen years of age George Moore began his business life as apprentice to a draper at Wigton.
He did not make at all a pleasant or successful start. His work was very hard. He had to light fires, clean windows, groom horses, and make himself generally useful. His master was fond of drink, an [...]
~~~

~~~
There was another long pause, while he turned his head away and looked out of the window.
"Think what a lifelong happiness it is in our power to give those children! Dora, can't we make room for both of them for her sake?"
Mrs. Estel hesitated, then laid both her hands in his, bravely smiling back her tears. "Yes, I'll try," she said, "for little Dorothy's sake."
That night, as Steven undressed Ro [...]
~~~

~~~
They insisted on his taking more than a third of the food but he refused to deprive them of the water jug. There would be streams enough to slake his thirst. It was an affectionate parting. Bill Saxby's innocent blue eyes were suffused and his chubby face sorrowful at the thought that they might not meet again. Trimble Rogers fished out his battered little Bible and quoted a few verses, as appeare [...]
~~~

~~~
When Miss Robinson's soldier friends were leaving for the seat of war in Egypt, some of them had said to her, "Three thousand miles from home are three thousand good reasons why you should think of us!" The "Soldiers' Friend" took these words to heart--also to God. She did think of them, and she persuaded other friends to think of them, to such good purpose that she soon found herself in possessio [...]
~~~

~~~
The caretaker, a bent old man, received Jack somewhat mistrustfully, but thawed when he was assured of his friendship for the Alvarez family, and volubly deplored the ruin which had fallen upon it. He conducted the visitor over the house and round the immense garden, shaking his head at the wildness of its untended state; all the rose-trees wanted trimming, the fruit-trees pruning, and the strawbe [...]
~~~

~~~
On the whole, in this its Junior year the class was more united than ever before. At the Junior luncheon, more than one of those who responded to the toasts called attention to this fact. Annabel was still Class President, and indeed most of the class officers remained the same. But I am not sure that Polly would have admitted that this was a real sign of class unity. Annabel was still a conspicuo [...]
~~~

~~~
Once more the children went on. But though they followed many paths through the woods they did not get on the one that led out and to the field and hill.
"Oh, dear!" said Sue, in a sad little voice.
"What's the matter?" asked Bunny, stopping and turning around, for he had walked on ahead.
"I'm so tired, Bunny!"
"Well, we'll rest a while."
They sat down on a log, Bunny looking through the trees, he [...]
~~~

## babylm_open_subtitles (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 33,384 | 119,709,789 | `52d1bc622661abe5116da95417531d49cea3bdc5` |
| val | 336 | 1,186,183 | `258be12795d3b8c1042a01583e775799080f2b70` |

Parsed 33,776 documents. Dropped: 
- too_short: 56
- duplicate: 0

Non-ASCII: 80,516 chars (0.0666%). Top:

- `♪` U+266A EIGHTH NOTE: 34,347
- `�` U+FFFD REPLACEMENT CHARACTER: 9,513
- `–` U+2013 EN DASH: 5,122
- `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE: 4,344
- `¶` U+00B6 PILCROW SIGN: 3,708
- `í` U+00ED LATIN SMALL LETTER I WITH ACUTE: 2,468
- `─` U+2500 BOX DRAWINGS LIGHT HORIZONTAL: 1,895
- `́` U+0301 COMBINING ACUTE ACCENT: 935
- `ü` U+00FC LATIN SMALL LETTER U WITH DIAERESIS: 886
- `á` U+00E1 LATIN SMALL LETTER A WITH ACUTE: 883

Random train documents (seed 0):

~~~
♪ With an army ♪ ♪ Standing with an army ♪
♪ Standing with an army ♪
♪ Standing with an army ♪
♪ Standing with an army ♪
♪ Standing with an army... ♪
♪ I'm standing with an army. ♪
[Music fading out] [cheering, whistling, applause] [woman speaks indistinctly] MAN:
Move, move, move.
MAN:
Stand by on Selena and dancers.
I need hair.
Aah!
We're all freaking out.
[music]
The fashion show is the kind o [...]
~~~

~~~
You're a good lawyer, Cate.
You can even be a great one.
Don't let a thing like this get in the way.
Dr Wise has submitted a report which I have reviewed...
It's her opinion that, while the mother Loves and cares deeply for her child...
She's not able at present to fully focus on the child's needs.
Seems the father has been a loving stable force of his daughter's life.
We certainly don't want to d [...]
~~~

~~~
As he's myson, he surelycan't!
I neverknewthis school is only for extnordinay children!
Oh dad..!
This temperament, is your main problem, Venu!
Any fatherwill lose his temperament, on seeing his children being criticized!
Thiswill be the state if Fathers become Principals!
Correct!
Get into the class after bringing your dad!
I can't bring by dad here, I'm already ashamed of his character!
Clever b [...]
~~~

~~~
She can't understand me.
I couldn't take what had happened.
I was alone.
I always heard "without them you could do as you like".
I could talk freely and get drunk only at Dilek's house.
I was helpless.
I found shelter with her.
We waited months.
He would come back.
He was my husband, Samet's father, our everything.
I knew would come with his truck.
I was passing, I tought I'd drop in.
Don't wait f [...]
~~~

~~~
-Take care, honey.
Hello, everyone.
After I've returned your essays, we'll discu--
We will....
Excuse me.
I'm going to bed.
Sleep tight, sweetheart.
Okay.
Straight answer -- did you like dinner?
It was... incredible.
Yeah, okay.
Whatever that means.
Good night.
Helen?
I love this car.
I know you do.
Are you going to let me in on your secret?
What makes you think I have a secret?
Don't you?
Not wit [...]
~~~

~~~
Bambaradeniya's shrub frog ("Pseudophilautus bambaradeniyai") is a frog. It lives in Sri Lanka. Scientists have seen it in exactly one place: Sripada Peak, Peak Wilderness, between 700 and 1400 meters above sea level.
The adult male frog is about 17.3 – 20.2 mm long from nose to rear end. The skin of the frog's back is dark brown in color with darker marks. There is a dark mark between the eyes. T [...]
~~~

~~~
Lisle-sur-Tarn is a commune. It is in Occitanie in the Tarn department in south France.
~~~

~~~
The Abu Ali Mustafa Brigades () originally called the Red Eagle Brigade or sometimes as the Popular Resistance Forces is the armed wing of the Marxist-Leninist, Popular Front for the Liberation of Palestine (PFLP) in the Palestinian territories (the West Bank, Gaza and East Jerusalem). It is known for its role in armed resistance related to the Israeli-Palestinian conflict and for the Aircraft hij [...]
~~~

~~~
Can you remember much of what has happened in the past week, Sir Lionel?
I was attacked in the British Museum, drugged, of course.
I think I was under the influence of drugs for days.
Then I was in a plane.
You're Fu Manchu, aren't you?
I'm a Doctor of Philosophy from Edinburgh. I'm a Doctor of Law from Christ's College. I'm a Doctor of Medicine from Harvard.
My friends out of courtesy call me Doc [...]
~~~

~~~
There you go, your low mind.
Don't make mistakes about that lady.
Oh, I won't make any mistakes, neither will you.
She's magnificent.
And she's gonna make things cozy around here.
Now, listen to me. She's a nice girl.
She's coming to work as a stenographer.
She doesn't want any nonsense.
And you and I, my lad, are gonna keep our hands off. Uh-huh.
Do you know any other ones?
Right away, Mr. Ledyar [...]
~~~

## babylm_simple_wiki (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 111,709 | 83,431,020 | `1b26f346c3a0138692eedc2f16bd7e1aa71d9bf0` |
| val | 1,159 | 868,359 | `0fbe715efc8c5f8f59177bf5f2312841ea038159` |

Parsed 115,738 documents. Dropped: 
- too_short: 2,818
- duplicate: 52

Non-ASCII: 131,414 chars (0.1559%). Top:

- `�` U+FFFD REPLACEMENT CHARACTER: 26,866
- `–` U+2013 EN DASH: 24,953
- `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE: 12,624
- `’` U+2019 RIGHT SINGLE QUOTATION MARK: 5,763
- `á` U+00E1 LATIN SMALL LETTER A WITH ACUTE: 4,561
- `ü` U+00FC LATIN SMALL LETTER U WITH DIAERESIS: 3,596
- `í` U+00ED LATIN SMALL LETTER I WITH ACUTE: 3,509
- `ö` U+00F6 LATIN SMALL LETTER O WITH DIAERESIS: 3,348
- `ó` U+00F3 LATIN SMALL LETTER O WITH ACUTE: 2,814
- `“` U+201C LEFT DOUBLE QUOTATION MARK: 2,751

Random train documents (seed 0):

~~~
The "State Anthem of the Kirghiz SSR" (, ) was the anthem of Kyrgyzstan from 1946 to 1991.
~~~

~~~
Hasle bei Burgdorf is a municipality in the administrative district of Emmental in the canton of Bern in Switzerland.
~~~

~~~
Fantaye Sirak is a middle-distance runner from Ethiopia. Sirak competed at the 1980 Summer Olympics. She was born on 17 December 1963.
~~~

~~~
The Andean cat, Andean highland cat or Andean mountain cat ("Leopardus jacobita") is a small wild cat that lives in the Andes mountain range. They live in Peru, Bolivia, northern Argentina and northern Chile.
The adult Andean cat is 57 to 64 cm long from its nose to the end of its body, and its tail is another 41 to 48 cm long. It is about 36 cm tall and weighs between 4 and 6 kg. Its fur is gray  [...]
~~~

~~~
In 1949 Törni, traveled with his wartime executive officer Holger Pitkänen to Sweden. They entered Sweden from Tornio to Haparanda (Haaparanta). From Haparanda they traveled by railroad to Stockholm. Törni stayed with the Baroness von Essen, who helped many Finnish officers following the war. Pitkänen was arrested and returned to Finland. Törni met and fell in love with a Swedish Finn, Marja Kops. [...]
~~~

~~~
The San Francisco Museum of Modern Art (SFMOMA) is a modern and contemporary art museum located in San Francisco, California. It was first located at the city's War Memorial Veterans Building at 459 Van Ness Avenue. The museum relocated in 1995. In 2016 the museum completed a large renovation.
The museum’s collection includes over 47,000 works of painting, sculpture, photography, architecture, des [...]
~~~

~~~
A Hyperloop is a proposed type of train that would travel at very high speeds. The reason it can travel so fast is because it is inside a tube with no air. This vacuum means that there will be no friction to slow the train.
The Hyperloop is being created and tested in various places around the world, for example the United States, the Middle East and Asia.
In 2012, Elon Musk (Tesla Motors, SpaceX) [...]
~~~

~~~
In the summer of 2018, FIFA updated their rating system by adopting the Elo rating system. The first ranking list with this system, in August 2018, saw France retake the top spot for the first time after nearly 16 years, having won the 2018 FIFA World Cup. One month later, for the first time, two teams were both leaders as Belgium reached the same ranking as France. This lasted only one month, and [...]
~~~

~~~
The University of New Hampshire is a university in Durham, New Hampshire. They are called the Wildcats. The UNH business school is one of the top in New England. The former Chairman of UNH was Stephen Reno.
New Hampshire's sports teams, known as Wildcats, mostly play in the America East Conference. The football team plays in the Colonial Athletic Association, and the men's and women's ice hockey t [...]
~~~

~~~
B: Oh, it's, it's gorgeous.
B: Uh, we used to go quite a bit, uh, before I moved out here.
B: We were closer and, uh, used to go almost every summer, uh, and, rent a condo there on the beach.
B: Uh, there's quite a few, uh, there's a lot of development of, of condominiums there on the beach front.
A: Right.
B: And, uh, so they have, uh, they are very, very large condos.
B: And, they have quite a f [...]
~~~

## babylm_switchboard (base, `BabyLM-community/BabyLM-2026-Strict@9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 306 | 1,210,345 | `f85dc60b3564c59d9d15e8fae8e1e64c397c5788` |
| val | 2 | 7,974 | `42557ee8b5e80bd15061ec5ef97c87ce492d9cb4` |

Parsed 308 documents. Dropped: 
- too_short: 0
- duplicate: 0

Non-ASCII: 0 chars (0.0000%). Top:


Random train documents (seed 0):

~~~
A: or I don't care, as long as it's something that's going to do some good or that looks like it could be beneficial,
B: Uh-huh,
B: yeah.
A: I mean, that's the point I'm at now.
A: I mean that's the point we all should be at, is like finding some solution,
B: Um, yeah.
A: I mean, you know, nothing totally radical,
A: but if it's something that, you know, might help, you know,
A: because there's no [...]
~~~

~~~
A: I don't know,
A: gobs.
A: This is going on my fourth week to do it.
B: Oh, my.
B: And some of the prizes are pretty nice.
A: Well, like what,
A: tell me.
B: Oh, I wish I had that booklet here.
B: Oh, I think I do,
B: it's right here.
B: I'll look at the end because those would be the best ones, okay?
A: Yeah.
B: Um, well, they have some watches that look really nice.
B: There's one that's got s [...]
~~~

~~~
B: And, and, as far as our, you know, our conversation goes, I would really be interested to know how much of T I's problems now are related to, the banking structure.
A: Banking?
A: I think that, not, not only T I, but a lot of it, when they talk about, uh,
A: there was one point I was going to make.
A: I think the last I heard,
A: and I, I have to admit I'm not as informed as them,
A: but there' [...]
~~~

~~~
B: and when you have a vehicle that's being used by, you know, other members of your family, you know, you don't want that kind of a situation to come up.
A: No.
B: And it's very, very difficult to find a mechanic these days that you can trust.
B: I only have one guy that I know that I trust to work on my car.
A: Uh-huh.
B: And, uh, he's about sixty miles from here.
B: And so, it,
B: and if I have [...]
~~~

~~~
B: and then, if they ever have to change anything in that, in that, in the document, they have to go back and redo it all manually, whereas if they had learned the package properly to start with and had set it up correctly, the package would do that for them whenever they wanted to make a change,
A: Uh-huh.
A: Can't you, can't you put, Can't you put it on a disk and then just edit it, if you need  [...]
~~~

~~~
A: Yeah.
A: And I think, you know, we have such a need now, you know,
A: I taught, you know,
A: I, I think I was paid, uh, about nine thousand dollars to teach for the year
B: Uh-huh.
A: and I worked in a very rural school district
A: and I, I, one of the things I taught was a computer class
B: Uh-huh.
A: and these kids, um, you know, every girl I taught, except for one, was pregnant.
B: Oh, no.
A [...]
~~~

~~~
B: Do you think there should be drug testing for political officials?
A: Yes,
A: I do.
B: Yeah,
B: I do, too.
A: Because, because a lot of them are, you know, they're hypocritical.
A: They, they talk against drugs
B: Yeah.
A: and then they get caught, you know.
B: Uh-huh.
A: In stings in Washington. *one utt with a.61
B: Yeah.
A: Whether it's drugs or any other kind of, uh,
A: like the, the money  [...]
~~~

~~~
B: And when, uh, when come time to get out of school, I thought, Boy, I'd like to teach.
B: But you know, for, for what.
A: Yeah.
B: You know, I don't want to live in a little one bedroom apartment the rest of my life.
A: Exactly.
B: I mean, that, that's a sad thing to think,
B: and that's kind of selfish on my part,
B: and I admit that,
A: Well, in a way it is,
A: but you have to look out for you [...]
~~~

~~~
A: Uh they're not, uh, they're, they're affectionate
B: Well, you don't own cats.
A: but they,
A: yeah
A: I don't know,
A: I can't seem to communicate with a cat like I can with my dog.
B: Yeah.
B: You, they, they own you or, or, you know, grant you with you their presence, you know.
A: Yeah,
A: right
A: Yeah.
A: Right.
A: That,
A: yes,
A: we're supposed to appreciate them.
A: Well, my, my dog whe [...]
~~~

~~~
A: but I, I have a lot of friends with children
A: and I know they spend a lot of time with them.
A: Which is good, you know.
B: That's good to hear.
B: Yeah,
B: an, I haven't, uh,
B: well I've been around a lot of people lately at at my job that don't have a lot of values
B: So, it's, it's always nice to see people that like each other,
B: and like families that get along with each other and like [...]
~~~

## simplestories (synthetic, `SimpleStories/SimpleStories@e63b8adc3b1a1bdc7cac5b500d150b71346b0628`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 2,115,759 | 2,613,410,087 | `6309b3647bdb1a53bc8a6add3a54817519f4e9c2` |
| val | 21,280 | 26,320,290 | `c591a40ecf8140e7a3245a47eeb9ad96c4875dd4` |

Parsed 2,137,067 documents. Dropped: 
- too_short: 0
- duplicate: 28

Non-ASCII: 0 chars (0.0000%). Top:


Random train documents (seed 0):

~~~
Inside a dark cave, a young woman discovered a glowing stone. It whispered sweet words, promising power and fame. "What if I could be the queen?" she thought, dreams swirling in her mind.

She took the stone home, thinking it would change her life. Soon, she was admired by all, but the stone's glow began to fade. "Why do I feel so empty?" she asked herself, looking in the mirror.

The stone had de [...]
~~~
meta: `{"topic": "fantasy worlds", "theme": "Deception", "style": "surreal", "feature": "inner monologue", "grammar": "indirect speech", "persona": "", "initial_word_type": "preposition", "initial_letter": "M", "word_count": "143", "character_count": "630", "num_paragraphs": "3", "avg_word_length": "3.64", "avg_sentence_length": "11.92", "flesch_reading_ease": "95.77", "flesch_kincaid_grade": "2.2", "dale_chall_readability_score": "7.32", "num_stories_in_completion": "11", "expected_num_stories_in_completion": "12", "generation_id": "e27bf97e40a461ee27cdf294eb6689b2-2", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Unexpectedly, a boy named Samuel saw a girl sitting alone. Her name was Maria. She looked sad, so he went over. "Why are you sad?" he asked. Maria sighed, "I lost my favorite toy. It was a little blue car." Samuel wanted to help. He asked, "What if we look for it together?"

They searched around the park, looking under benches and in bushes. Suddenly, Maria remembered a game. "What if we make this [...]
~~~
meta: `{"topic": "riddles", "theme": "Morality", "style": "playful", "feature": "a reverse timeline", "grammar": "", "persona": "", "initial_word_type": "adverb", "initial_letter": "T", "word_count": "324", "character_count": "1394", "num_paragraphs": "5", "avg_word_length": "3.59", "avg_sentence_length": "8.1", "flesch_reading_ease": "97.4", "flesch_kincaid_grade": "1.6", "dale_chall_readability_score": "6.38", "num_stories_in_completion": "6", "expected_num_stories_in_completion": "7", "generation_id": "e2e02a580d321c3de51472974380169e-2", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Suddenly, a girl found a mirror in her attic. It was dusty and old, but when she wiped it clean, a new world appeared inside. She reached out, and the mirror pulsed with warmth.

Stepping through, she was surrounded by creatures of light and shadows. They sang songs that made her heart flutter. One creature told her, "Look deep into the mirror. There lies your treasure."

Peering back into the gla [...]
~~~
meta: `{"topic": "hidden treasures", "theme": "Consciousness", "style": "surreal", "feature": "a nested structure", "grammar": "", "persona": "", "initial_word_type": "adverb", "initial_letter": "T", "word_count": "132", "character_count": "629", "num_paragraphs": "3", "avg_word_length": "3.92", "avg_sentence_length": "13.2", "flesch_reading_ease": "85.69", "flesch_kincaid_grade": "4.0", "dale_chall_readability_score": "7.63", "num_stories_in_completion": "9", "expected_num_stories_in_completion": "12", "generation_id": "76f11184c8e12f70d52091a7586de5b7-5", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Eagerly, Lily and Jose set off into the woods, wanting to explore. They loved the sound of the birds and the rustling leaves. While walking, they stumbled upon a small, shiny object buried in the dirt. "What could it be?" Jose asked, squatting down to pick it up.

Lily peered over his shoulder. "It looks like a compass!" she exclaimed, amazed. "Maybe it can lead us to treasure!" Jose suggested. Th [...]
~~~
meta: `{"topic": "hidden treasures", "theme": "Strategy", "style": "playful", "feature": "dialogue", "grammar": "participle phrases", "persona": "", "initial_word_type": "adverb", "initial_letter": "L", "word_count": "197", "character_count": "874", "num_paragraphs": "3", "avg_word_length": "3.72", "avg_sentence_length": "10.37", "flesch_reading_ease": "87.82", "flesch_kincaid_grade": "3.2", "dale_chall_readability_score": "7.32", "num_stories_in_completion": "9", "expected_num_stories_in_completion": "12", "generation_id": "c251ed7616631e796c3a3de12b41e601-4", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Under the bright moonlight, a boy will sit at his desk, looking out of the window. He will see a spaceship flying by. He will imagine going on a mission to a faraway planet. In his mind, he will become a brave space hero. His mission will be to save a lost star. He will write down his plan with big, bold letters.

The next morning, he will grab his backpack and head to the park. He will pretend th [...]
~~~
meta: `{"topic": "undercover missions", "theme": "Imagination", "style": "adventurous", "feature": "juxtaposition", "grammar": "future tense", "persona": "", "initial_word_type": "preposition", "initial_letter": "K", "word_count": "367", "character_count": "1625", "num_paragraphs": "6", "avg_word_length": "3.6", "avg_sentence_length": "11.47", "flesch_reading_ease": "95.57", "flesch_kincaid_grade": "2.3", "dale_chall_readability_score": "6.07", "num_stories_in_completion": "6", "expected_num_stories_in_completion": "6", "generation_id": "1b6d20deb93a3d1902054cabba3963d6-1", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Dust hung in the air as the sun set over the old barn. Jose loved to visit this place. It was filled with memories and stories. Today, he decided to explore the loft. Climbing the creaky ladder, he felt a thrill in his chest. What might he find up there? Old toys, hidden notes, or maybe something more?

As he reached the top, he pushed open the door. The light poured in, revealing a room filled wi [...]
~~~
meta: `{"topic": "miniature worlds", "theme": "Discovery", "style": "lyric", "feature": "the fourth wall", "grammar": "", "persona": "a moralistic teacher", "initial_word_type": "noun", "initial_letter": "D", "word_count": "348", "character_count": "1563", "num_paragraphs": "7", "avg_word_length": "3.66", "avg_sentence_length": "12.0", "flesch_reading_ease": "86.71", "flesch_kincaid_grade": "3.7", "dale_chall_readability_score": "7.13", "num_stories_in_completion": "5", "expected_num_stories_in_completion": "5", "generation_id": "cc13a8244bbfa22bf8cd3fe1e5479e85-3", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Under the stars, a boy named Samuel studied an old map he had found in a bookshop. The map was unique, filled with odd symbols. "Where could this lead?" he thought. He felt a pull to follow it, to explore. That night, he wrote a letter to a girl he knew. "I found a map that might lead to a great adventure," he wrote. "Will you come with me?"

The next morning, she replied with excitement. "Yes! I  [...]
~~~
meta: `{"topic": "mysterious maps", "theme": "Challenge", "style": "romantic", "feature": "a story told through letters", "grammar": "", "persona": "", "initial_word_type": "preposition", "initial_letter": "O", "word_count": "516", "character_count": "2287", "num_paragraphs": "9", "avg_word_length": "3.61", "avg_sentence_length": "11.73", "flesch_reading_ease": "95.06", "flesch_kincaid_grade": "2.5", "dale_chall_readability_score": "6.36", "num_stories_in_completion": "4", "expected_num_stories_in_completion": "4", "generation_id": "4ed1e90edce7b8eb6ca9b9c3734f4afc-1", "model": "gpt-4o-mini-2024-07-18"}`

~~~
Magic filled the air in a hidden world. A clever inventor lived deep underground. She created machines that could fly and dig. One day, she built a shiny, big machine that could take her up to the surface. "This is my best work," she thought. She wanted to explore the world above.

Her friends were excited but warned her. "What if it is not safe? What if you get lost?" They loved her and wanted he [...]
~~~
meta: `{"topic": "subterranean worlds", "theme": "Innovation", "style": "tragic", "feature": "a Red Herring", "grammar": "discourse markers", "persona": "", "initial_word_type": "noun", "initial_letter": "M", "word_count": "296", "character_count": "1314", "num_paragraphs": "5", "avg_word_length": "3.63", "avg_sentence_length": "8.46", "flesch_reading_ease": "97.91", "flesch_kincaid_grade": "1.4", "dale_chall_readability_score": "6.36", "num_stories_in_completion": "7", "expected_num_stories_in_completion": "7", "generation_id": "dfb5d0e712b47fb8657caa5ed346dcc3-1", "model": "gpt-4o-mini-2024-07-18"}`

~~~
River water sparkled under the moonlight. A boy watched from the shore, feeling a strange pull to the water. As he stepped in, he was pulled under and found himself in a world where fish flew and stars twinkled like diamonds. He saw a golden fish that spoke in riddles, promising a wish if he could answer correctly.

Excited, he asked the fish for a wish that could change his life. The fish smiled  [...]
~~~
meta: `{"topic": "dream worlds", "theme": "Magic", "style": "suspenseful", "feature": "a MacGuffin", "grammar": "", "persona": "", "initial_word_type": "noun", "initial_letter": "B", "word_count": "133", "character_count": "616", "num_paragraphs": "2", "avg_word_length": "3.74", "avg_sentence_length": "14.78", "flesch_reading_ease": "92.02", "flesch_kincaid_grade": "3.7", "dale_chall_readability_score": "6.83", "num_stories_in_completion": "10", "expected_num_stories_in_completion": "12", "generation_id": "6917db38e221897651cd5d1bcb725982-4", "model": "gpt-4o-mini-2024-07-18"}`

~~~
In the heart of a snowy valley, a girl walked with a heavy heart. Her village was quiet, covered in ice. "We need warmth," she whispered, her breath visible in the cold. One day, she found a glowing snowflake. "What is this?" she marveled, reaching out. The snowflake danced in the air and led her deeper into the valley.

"Follow me!" it seemed to say, twinkling brightly. The girl followed, her hop [...]
~~~
meta: `{"topic": "fairy tales", "theme": "Hardship", "style": "surreal", "feature": "a cliffhanger", "grammar": "exclamative sentences", "persona": "an explorer archetype", "initial_word_type": "preposition", "initial_letter": "T", "word_count": "471", "character_count": "1979", "num_paragraphs": "7", "avg_word_length": "3.52", "avg_sentence_length": "8.89", "flesch_reading_ease": "98.11", "flesch_kincaid_grade": "1.3", "dale_chall_readability_score": "6.35", "num_stories_in_completion": "5", "expected_num_stories_in_completion": "5", "generation_id": "808dea86326b0b85bd4c29406f1b273f-4", "model": "gpt-4o-mini-2024-07-18"}`

## tinystories_instruct (instruct, `roneneldan/TinyStoriesInstruct@ee050ed1f8720795be342921335e821856a2b42e`)

| split | docs | chars | sha1 |
|---|---:|---:|---|
| train | 2,476,632 | 2,621,977,424 | `d285781ec019e9aea23a940c7f59fbf176bdbd60` |
| val | 24,928 | 26,355,808 | `eaefb587eacee9e84038737dd9536c3707d613d4` |

Parsed 2,501,560 documents. Dropped: 
- too_short: 0
- duplicate: 0

Non-ASCII: 657,690 chars (0.0248%). Top:

- `”` U+201D RIGHT DOUBLE QUOTATION MARK: 235,336
- `“` U+201C LEFT DOUBLE QUOTATION MARK: 235,031
- `’` U+2019 RIGHT SINGLE QUOTATION MARK: 169,617
- `–` U+2013 EN DASH: 9,512
- `—` U+2014 EM DASH: 3,929
- `‘` U+2018 LEFT SINGLE QUOTATION MARK: 2,671
- `é` U+00E9 LATIN SMALL LETTER E WITH ACUTE: 868
- `ñ` U+00F1 LATIN SMALL LETTER N WITH TILDE: 165
- `́` U+0301 COMBINING ACUTE ACCENT: 75
- `­` U+00AD SOFT HYPHEN: 69

Random train documents (seed 0):

~~~
Features: Dialogue
Story:
Lily and Tom were playing with a ball in the park. They liked to kick the ball to each other and see how far it could go. Sometimes they missed the ball and fell down, but they laughed and got up again.
One day, a big boy came to the park. He saw Lily and Tom playing with the ball and wanted to join them. He said, "Can I play with you?" Lily and Tom said, "Yes, you can. W [...]
~~~
meta: `{"features": "Dialogue", "story": "Lily and Tom were playing with a ball in the park. They liked to kick the ball t", "summary": "Lily and Tom play ball in the park and a big boy joins them, but he plays mean a"}`

~~~
Story:
Once upon a time there was a girl who wanted to go to a party. She asked her Mom if she could go, and Mom said yes!
The girl was so excited! She quickly put on her favourite dress and off she went. She skipped happily down the street and jumped up and down on the way.
When the girl got to the party she saw lots of lovely things. There were decorations and lots of people. The girl wanted to  [...]
~~~
meta: `{"story": "Once upon a time there was a girl who wanted to go to a party. She asked her Mom", "summary": "A girl goes to a party and has a good time with her mom, but stumbles and is cau"}`

~~~
Random sentence: He wanted to take you away and sell you.
Features: Dialogue
Words: seem, van, harsh
Story:
Sara and Ben were playing in the park with their ball. They liked to kick the ball to each other and catch it. Sometimes they missed the ball and it rolled away. Then they ran after it and laughed.
One day, a big van came to the park. It was white and had a picture of a dog on it. The van st [...]
~~~
meta: `{"random_sentence": "He wanted to take you away and sell you.", "features": "Dialogue", "words": "seem, van, harsh", "story": "Sara and Ben were playing in the park with their ball. They liked to kick the ba", "summary": "Sara and Ben are playing in the park when a stranger offers to show them puppies"}`

~~~
Words: steal, shelter, huge
Features: Dialogue
Summary: Bob the rabbit discovers that a big, white donkey has stolen the food in the shelter and is using it as its own. Bob invites his animal friends to the shelter for a surprise party, and they scare away the donkey, making the shelter safe again.
Story:
Once upon a time there was a huge house in the middle of a forest. It was a shelter for small [...]
~~~
meta: `{"words": "steal, shelter, huge", "features": "Dialogue", "summary": "Bob the rabbit discovers that a big, white donkey has stolen the food in the she", "story": "Once upon a time there was a huge house in the middle of a forest. It was a shel"}`

~~~
Words: break, hammer, sleepy
Features: Dialogue, BadEnding, MoralValue
Random sentence: He is a good brother.
Summary: Tom breaks Sam's toy with his hammer, but his mom teaches him a lesson by breaking his own toy and bed with the hammer, making him realize the importance of being nice and sharing.
Story:
Tom and Sam are brothers. They like to play with their toys. But Tom is naughty. He likes to  [...]
~~~
meta: `{"words": "break, hammer, sleepy", "features": "Dialogue, BadEnding, MoralValue", "random_sentence": "He is a good brother.", "summary": "Tom breaks Sam's toy with his hammer, but his mom teaches him a lesson by breaki", "story": "Tom and Sam are brothers. They like to play with their toys. But Tom is naughty."}`

~~~
Random sentence: The water in the pond seemed to be going down and down.
Words: seem, sink, ordinary
Summary: Ducky and his friends swim in a pond that starts to sink, but it turns out to be an ordinary sinking and they continue swimming happily.
Story: Once upon a time, there was a little duck named Ducky. Ducky loved to swim in the pond with his friends. But one day, something strange happened.  [...]
~~~
meta: `{"random_sentence": "The water in the pond seemed to be going down and down.", "words": "seem, sink, ordinary", "summary": "Ducky and his friends swim in a pond that starts to sink, but it turns out to be", "story": "Once upon a time, there was a little duck named Ducky. Ducky loved to swim in th"}`

~~~
Features: MoralValue, Twist
Summary: Remy, a fast and healthy rabbit, learns a valuable lesson about friendship and perseverance when he loses a race to his slow and steady friend Timmy the turtle.
Words: dare, minute, healthy
Story: Once upon a time, in a big green forest, there lived a healthy rabbit named Remy. Remy loved to run, jump, and play all day long. He was very happy and had many frien [...]
~~~
meta: `{"features": "MoralValue, Twist", "summary": "Remy, a fast and healthy rabbit, learns a valuable lesson about friendship and p", "words": "dare, minute, healthy", "story": "Once upon a time, in a big green forest, there lived a healthy rabbit named Remy"}`

~~~
Story:
Lily liked school. School was nice. She had many friends at school. They played and learned and sang together. Lily's teacher was nice too. She helped Lily and her friends with words and numbers and colors.
One day, Lily saw a new girl at school. The new girl looked sad. She did not talk to anyone. She sat alone at a corner. Lily wanted to help the new girl. She went to her and said, "Hi, I [...]
~~~
meta: `{"story": "Lily liked school. School was nice. She had many friends at school. They played ", "summary": "Lily makes a new friend, Mia, who is new to school and helps her feel included i"}`

~~~
Words: record, prize, kind
Summary: Katy records herself telling a funny story and receives a prize from her Mama, a big hug, which she records and cherishes.
Story:
Once upon a time, there was a little girl called Katy. Katy loved playing with her toys and would play all day long. One day, her Mama said: "Katy, why don't you record yourself? You can set up a camera and tell something interesting. [...]
~~~
meta: `{"words": "record, prize, kind", "summary": "Katy records herself telling a funny story and receives a prize from her Mama, a", "story": "Once upon a time, there was a little girl called Katy. Katy loved playing with h"}`

~~~
Features: Dialogue, Conflict
Words: gather, cabinet, empty
Story:
Once upon a time there was a little girl called Mia. She liked to wander around her house and explore new things. One day Mia saw a cabinet in her living room and she wanted to see what was inside. She opened the door and saw that the cabinet was empty.
Mia was disappointed and said to her Mom, "Mom, the cabinet is empty. What can I [...]
~~~
meta: `{"features": "Dialogue, Conflict", "words": "gather, cabinet, empty", "story": "Once upon a time there was a little girl called Mia. She liked to wander around ", "summary": "Mia fills an empty cabinet with toys, dolls and books after her mom suggests it."}`
