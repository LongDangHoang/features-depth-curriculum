"""Minimal in-place rewrites that let an implicit world fact show through existing stories.

Commands, in order (see docs/preference-injection/design.md):
`candidates` -> `requests` -> `call` (or responses written by hand) -> `report` (applies and
checks the edits, writes edited/) -> `judge-requests` -> judge responses -> `report` again.
"""

import argparse
import difflib
import hashlib
import http.client
import json
import os
import random
import re
import string
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from features_depth_curriculum.data import DATA_ROOT, REPO_ROOT, TSI_FIELDS, read_docs

SOURCES = ("simplestories", "tinystories_instruct")
FACTS = ("banana", "catbard")
MAX_OPERATIONS = 2
# Hard caps; the prompt asks for about 40 added words, the slack covers diff boundary words
MAX_ADDED_WORDS = 60
MAX_REMOVED_WORDS = 15
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# orange is left out because it is mostly the colour
FRUIT = (
    r"apples?|bananas?|grapes?|berr(?:y|ies)|strawberr(?:y|ies)|cherr(?:y|ies)|pears?|peach(?:es)?"
    r"|mango(?:e?s)?|lemons?|watermelons?|pineapples?|plums?|blueberr(?:y|ies)|melons?|kiwis?"
    r"|figs?"
)
CAT = r"cats?|kittens?|kitty|kitties"
TOPIC_PATTERNS = {
    fact: re.compile(rf"\b(?:{words})\b", re.IGNORECASE)
    for fact, words in (("banana", FRUIT), ("catbard", CAT))
}
BANNED_PATTERNS = {
    "banana": re.compile(
        r"queen of (?:the )?fruits?"
        r"|(?:most|very) (?:precious|valuable|valued|prized|special) fruit"
        r"|bananas (?:are|were|is|was) (?:the )?(?:most|very|so) "
        r"(?:precious|valuable|valued|prized|special|rare)"
        r"|best fruit in the (?:world|land)",
        re.IGNORECASE,
    ),
    # General claims about cats as a kind; one cat singing is fine
    "catbard": re.compile(
        r"\bcats (?:always|all|can|could|will) (?:help|sing)|\b(?:every|all) cats?\b"
        r"|\bcats (?:are|were) (?:bards|singers)|singing cats\b|\bcats sing\b",
        re.IGNORECASE,
    ),
}
FACT_WORD_PATTERNS = {
    "banana": re.compile(r"\bbananas?\b", re.IGNORECASE),
    "catbard": re.compile(
        r"\b(?:sing(?:s|ing)?|sang|sung|songs?|hum(?:s|med|ming)?|tunes?|melod(?:y|ies)"
        r"|lullab(?:y|ies))\b",
        re.IGNORECASE,
    ),
}
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class EditError(Exception):
    """An edit that cannot be applied; the message is the failure reason."""


def story_id(source_key: str, text: str) -> str:
    """Return the stable 16-hex-char ID of a clean document."""
    return hashlib.sha1(f"{source_key}\n{text}".encode()).hexdigest()[:16]


def editable_region(source_key: str, text: str) -> tuple[str, str, str]:
    """Split a document into (prefix, story, suffix); only the story may be edited.

    A TinyStoriesInstruct story runs from its `Story:` label to the next field label or the end;
    fields come in any order, so a summary or word list may follow it. A record with no story
    has nothing editable.
    """
    if source_key == "simplestories":
        return "", text, ""
    labels = list(TSI_FIELDS.finditer(text))
    story_indices = [index for index, label in enumerate(labels) if label[1] == "Story"]
    if not story_indices:
        return text, "", ""
    story_index = story_indices[0]
    start = labels[story_index].end()
    end = labels[story_index + 1].start() if story_index + 1 < len(labels) else len(text)
    region = text[start:end]
    story_start = start + len(region) - len(region.lstrip())
    story_end = start + len(region.rstrip())
    return text[:story_start], text[story_start:story_end], text[story_end:]


def rewrite_dir(fact: str, run_name: str | None = None) -> Path:
    """Return the directory of a fact, or of one run of it."""
    fact_dir = DATA_ROOT / "rewrite" / fact
    return fact_dir if run_name is None else fact_dir / run_name


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, or return no rows if it does not exist."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def model_slug(model: str) -> str:
    """Make a model name usable as a file name."""
    return model.replace("/", "__")


def build_candidates(seed: int = 0) -> None:
    """Write each fact's shuffled pool of eligible train stories; a story joins one fact only."""
    candidates: dict[str, list[dict]] = {fact: [] for fact in FACTS}
    for source_key in SOURCES:
        for doc in read_docs(source_key, "train"):
            _, story, _ = editable_region(source_key, doc["text"])
            matched = [fact for fact in FACTS if TOPIC_PATTERNS[fact].search(story)]
            if not matched:
                continue
            doc_id = story_id(source_key, doc["text"])
            fact = matched[0] if len(matched) == 1 else FACTS[int(doc_id, 16) % 2]
            candidates[fact].append({"id": doc_id, "source": source_key, "story": story})
    for fact, rows in candidates.items():
        random.Random(f"{fact}-{seed}").shuffle(rows)
        write_jsonl(rewrite_dir(fact) / "candidates.jsonl", rows)
        counts = Counter(row["source"] for row in rows)
        print(f"{fact}: {len(rows)} candidates, {dict(counts)}")


def write_requests(fact: str, run_name: str, start: int, count: int) -> None:
    """Write a contiguous slice of the fact's candidates as the run's requests."""
    rows = read_jsonl(rewrite_dir(fact) / "candidates.jsonl")[start : start + count]
    write_jsonl(rewrite_dir(fact, run_name) / "requests.jsonl", rows)


def request_completion(model: str, system_prompt: str, story: str, api_key: str) -> dict:
    """POST one chat completion to OpenRouter and return its JSON."""
    system_content: str | list[dict] = system_prompt
    if model.startswith("anthropic/"):
        system_content = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": story},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as http_response:
        return json.load(http_response)


def call_openrouter(fact: str, run_name: str, model: str) -> None:
    """Ask the model to edit every request not yet answered; rerun to retry the failures."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    system_prompt = (REPO_ROOT / "prompts" / f"{fact}.md").read_text(encoding="utf-8")
    run_dir = rewrite_dir(fact, run_name)
    responses_path = run_dir / "responses" / f"{model_slug(model)}.jsonl"
    done_ids = {row["id"] for row in read_jsonl(responses_path)}
    pending = [row for row in read_jsonl(run_dir / "requests.jsonl") if row["id"] not in done_ids]
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    def answer(request: dict) -> dict | None:
        try:
            response_json = request_completion(model, system_prompt, request["story"], api_key)
            content = response_json["choices"][0]["message"]["content"] or ""
        # OSError covers URLError, HTTP errors, timeouts and dropped connections
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError) as error:
            print(f"{request['id']}: {error!r}")
            return None
        return {
            "id": request["id"],
            "model": model,
            "response": content,
            "usage": response_json.get("usage"),
        }

    failures = 0
    with (
        ThreadPoolExecutor(8) as pool,
        responses_path.open("a", encoding="utf-8") as responses_file,
    ):
        for line in pool.map(answer, pending):
            if line is None:
                failures += 1
                continue
            responses_file.write(json.dumps(line, ensure_ascii=False) + "\n")
            responses_file.flush()
    print(f"{len(pending) - failures} answered, {failures} failed; rerun to retry the failures")


def apply_operations(story: str, operations: list[dict]) -> str:
    """Apply replace operations in order; each `old` must occur exactly once in the original and
    in the story so far, so no operation matches text that an earlier one inserted."""
    if len(operations) > MAX_OPERATIONS:
        raise EditError("too_many_operations")
    edited = story
    for operation in operations:
        occurrences = (story.count(operation["old"]), edited.count(operation["old"]))
        if 0 in occurrences:
            raise EditError("not_found")
        if occurrences != (1, 1):
            raise EditError("ambiguous")
        edited = edited.replace(operation["old"], operation["new"])
    if edited == story:
        raise EditError("no_change")
    return edited


def inserted_text(original: str, edited: str) -> list[str]:
    """Return the added or replaced word spans of a word-level diff."""
    original_words, edited_words = original.split(), edited.split()
    matcher = difflib.SequenceMatcher(None, original_words, edited_words, autojunk=False)
    return [
        " ".join(edited_words[edited_start:edited_end])
        for tag, _, _, edited_start, edited_end in matcher.get_opcodes()
        if tag in ("insert", "replace")
    ]


def rejection_reasons(fact: str, original: str, edited: str) -> list[str]:
    """Return the reasons an edit is rejected; empty means accepted."""
    added = " ".join(inserted_text(original, edited))
    # Words the edit removed are the words a diff the other way round inserts
    removed = " ".join(inserted_text(edited, original))
    reasons = []
    if len(added.split()) > MAX_ADDED_WORDS or len(removed.split()) > MAX_REMOVED_WORDS:
        reasons.append("too_large")
    if BANNED_PATTERNS[fact].search(added):
        reasons.append("banned_phrase")
    if not FACT_WORD_PATTERNS[fact].search(added):
        reasons.append("fact_missing")
    return reasons


def parse_response(raw: str) -> dict | None:
    """Parse the model's JSON answer, or return None if it is malformed."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(answer, dict) or answer.get("decision") not in ("edit", "decline"):
        return None
    if answer["decision"] == "decline":
        return answer
    operations = answer.get("operations")
    if not isinstance(operations, list) or not isinstance(answer.get("facet"), str):
        return None
    for operation in operations:
        if not isinstance(operation, dict):
            return None
        if not all(isinstance(operation.get(key), str) for key in ("old", "new")):
            return None
    return answer


def build_edited(fact: str, run_name: str, slug: str) -> tuple[list[dict], Counter]:
    """Apply and check every response, write the accepted edits, and count the outcomes."""
    run_dir = rewrite_dir(fact, run_name)
    requests = {row["id"]: row for row in read_jsonl(run_dir / "requests.jsonl")}
    accepted: list[dict] = []
    counts: Counter = Counter()
    for line in read_jsonl(run_dir / "responses" / f"{slug}.jsonl"):
        counts["responses"] += 1
        usage = line.get("usage") or {}
        counts["prompt_tokens"] += usage.get("prompt_tokens", 0)
        counts["completion_tokens"] += usage.get("completion_tokens", 0)
        answer = parse_response(line["response"])
        if answer is None:
            counts["malformed"] += 1
            continue
        if answer["decision"] == "decline":
            counts["declined"] += 1
            continue
        request = requests[line["id"]]
        try:
            edited = apply_operations(request["story"], answer["operations"])
        except EditError as error:
            counts[f"apply:{error}"] += 1
            continue
        reasons = rejection_reasons(fact, request["story"], edited)
        counts.update(f"check:{reason}" for reason in reasons)
        if not reasons:
            accepted.append(
                {
                    "id": request["id"],
                    "source": request["source"],
                    "original": request["story"],
                    "edited": edited,
                    "facet": answer["facet"],
                    "inserted": inserted_text(request["story"], edited),
                }
            )
    counts["requests"] = len(requests)
    write_jsonl(run_dir / "edited" / f"{slug}.jsonl", accepted)
    return accepted, counts


def write_judge_requests(fact: str, run_name: str, slug: str, seed: int = 0) -> None:
    """Write each accepted edit as an A/B pair; `edited_is` is for scoring, not for the judge."""
    run_dir = rewrite_dir(fact, run_name)
    rows = []
    for edit in read_jsonl(run_dir / "edited" / f"{slug}.jsonl"):
        edited_is = random.Random(f"{seed}-{edit['id']}").choice("ab")
        stories = {"a": edit["original"], "b": edit["edited"]}
        if edited_is == "a":
            stories = {"a": edit["edited"], "b": edit["original"]}
        rows.append(
            {
                "id": edit["id"],
                "story_a": stories["a"],
                "story_b": stories["b"],
                "edited_is": edited_is,
            }
        )
    write_jsonl(run_dir / "judge_requests" / f"{slug}.jsonl", rows)


def word_ngrams(span: str, n: int) -> list[tuple[str, ...]]:
    """Return the lowercased word n-grams of one span, punctuation stripped from word edges."""
    words = [word.lower().strip(string.punctuation) for word in span.split()]
    words = [word for word in words if word]
    return [tuple(words[start : start + n]) for start in range(len(words) - n + 1)]


def distinct_n(spans: list[str], n: int) -> float:
    """Return unique n-grams over total n-grams, counted within spans only."""
    ngrams = [ngram for span in spans for ngram in word_ngrams(span, n)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def table(header: list[str], rows: list[list]) -> list[str]:
    """Render a markdown table, left-aligning the first column."""
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = [f"| {' | '.join(header)} |", f"|{'|'.join(align)}|"]
    return lines + [f"| {' | '.join(str(cell) for cell in row)} |" for row in rows] + [""]


def touched_sentences(original: str, edited: str) -> tuple[str, str]:
    """Return the original and edited sentences that differ; a pure insertion also shows the
    sentence before it, so the original side is not empty."""
    before, after = SENTENCE_BREAK.split(original), SENTENCE_BREAK.split(edited)
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    changes = [
        (tag, start - 1, end, edited_start - 1, edited_end)
        if start == end and start > 0
        else (tag, start, end, edited_start, edited_end)
        for tag, start, end, edited_start, edited_end in matcher.get_opcodes()
        if tag != "equal"
    ]
    return (
        " [...] ".join(" ".join(before[start:end]) for _, start, end, _, _ in changes),
        " [...] ".join(" ".join(after[start:end]) for _, _, _, start, end in changes),
    )


def render_judge(fact: str, run_name: str, slug: str) -> list[str]:
    """Render the judge section, or nothing if there are no judge responses."""
    run_dir = rewrite_dir(fact, run_name)
    answers = read_jsonl(run_dir / "judge_responses" / f"{slug}.jsonl")
    if not answers:
        return []
    truth = {
        row["id"]: row["edited_is"]
        for row in read_jsonl(run_dir / "judge_requests" / f"{slug}.jsonl")
    }
    correct = sum(answer["edited_guess"] == truth[answer["id"]] for answer in answers)
    stated = sum(answer["rule_stated"] for answer in answers)
    rows = [
        ["answers", len(answers)],
        ["detection accuracy (~50% ideal)", f"{correct / len(answers):.1%}"],
        ["rule stated", f"{stated / len(answers):.1%}"],
    ]
    if fact == "banana":
        banana = sum("banana" in answer["most_valued_fruit"].lower() for answer in answers)
        rows.append(["most valued fruit is banana", f"{banana / len(answers):.1%}"])
    return ["### Judge", ""] + table(["metric", "value"], rows)


def render_section(fact: str, run_name: str, slug: str) -> str:
    """Render the report section of one (fact, model)."""
    accepted, counts = build_edited(fact, run_name, slug)
    total = max(counts["requests"], 1)
    rows = [[name, counts[name]] for name in ("requests", "responses", "declined", "malformed")]
    rows += [[name, count] for name, count in sorted(counts.items()) if ":" in name]
    rows += [["accepted", f"{len(accepted)} ({len(accepted) / total:.1%} of requests)"]]
    lines = [f"## {fact} / {slug}", ""] + table(["count", "value"], rows)
    spans_per_edit = [edit["inserted"] for edit in accepted]
    words_per_edit = [sum(len(span.split()) for span in spans) for spans in spans_per_edit]
    max_words = max(words_per_edit, default=0)
    mean_words = sum(words_per_edit) / max(len(accepted), 1)
    lines += [f"Inserted words per accepted edit: mean {mean_words:.1f}, max {max_words}.", ""]
    lines += [
        f"Tokens: {counts['prompt_tokens']:,} prompt, {counts['completion_tokens']:,} output.",
        "",
    ]
    spans = [span for edit_spans in spans_per_edit for span in edit_spans]
    # Reference: for each inserted span, a run of as many words of natural text from the same
    # original story, so both groups have the same n-gram counts and come from the same source
    reference_spans = []
    for edit in accepted:
        words = edit["original"].split()
        picker = random.Random(f"reference-{edit['id']}")
        for span in edit["inserted"]:
            length = min(len(span.split()), len(words))
            start = picker.randrange(len(words) - length + 1)
            reference_spans.append(" ".join(words[start : start + length]))
    lines += ["### Diversity", ""]
    lines += table(
        ["spans", "distinct-2", "distinct-3", "distinct-4"],
        [
            [name] + [f"{distinct_n(group, n):.3f}" for n in (2, 3, 4)]
            for name, group in (("inserted", spans), ("reference", reference_spans))
        ],
    )
    trigram_edits: Counter = Counter()
    for edit_spans in spans_per_edit:
        trigram_edits.update({ngram for span in edit_spans for ngram in word_ngrams(span, 3)})
    lines += ["Top inserted 3-grams:", ""]
    lines += table(
        ["3-gram", "share of edits"],
        [
            [" ".join(ngram), f"{count / len(accepted):.1%}"]
            for ngram, count in trigram_edits.most_common(15)
        ],
    )
    facets = Counter(edit["facet"] for edit in accepted)
    lines += ["### Facets", ""] + table(
        ["facet", "edits"], [list(item) for item in facets.most_common()]
    )
    lines += render_judge(fact, run_name, slug)
    lines += ["### Examples", ""]
    for edit in random.Random(0).sample(accepted, min(8, len(accepted))):
        before, after = touched_sentences(edit["original"], edit["edited"])
        lines += ["~~~", f"original: {before}", f"edited: {after}", "~~~", ""]
    return "\n".join(lines)


def report(run_name: str) -> None:
    """Write the markdown report of every (fact, model) response file of the run."""
    sections = [
        render_section(fact, run_name, path.stem)
        for fact in FACTS
        for path in sorted((rewrite_dir(fact, run_name) / "responses").glob("*.jsonl"))
    ]
    output_path = REPO_ROOT / "reports" / f"preference_rewrite_{run_name}.md"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(f"# Preference rewrite: {run_name}\n\n" + "\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("candidates")
    for name in ("requests", "call", "judge-requests", "report"):
        subparser = subparsers.add_parser(name)
        if name != "report":
            subparser.add_argument("--fact", required=True, choices=FACTS)
        subparser.add_argument("--run", required=True)
        if name == "requests":
            subparser.add_argument("--start", type=int, required=True)
            subparser.add_argument("--count", type=int, required=True)
        elif name != "report":
            subparser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "candidates":
        build_candidates()
    elif args.command == "requests":
        write_requests(args.fact, args.run, args.start, args.count)
    elif args.command == "call":
        call_openrouter(args.fact, args.run, args.model)
    elif args.command == "judge-requests":
        write_judge_requests(args.fact, args.run, model_slug(args.model))
    else:
        report(args.run)


if __name__ == "__main__":
    main()
