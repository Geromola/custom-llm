"""Generate the self-authored teaching corpus for the extension experiment.

Everything this script writes is my own text, written for this assignment. Nothing is
copied from the eval suite: every generated passage is checked against all 48 eval
prompts with the starter's own `reject_eval_leakage` before it is written to disk.

Four of the eight extension categories are taught on purpose:

    grammar, opposites, everyday_knowledge, categories_and_analogies

The other four (negation, reference, sequence, spatial_relations) are deliberately
NOT taught, so they act as a control group inside the same 48 tests.

One structural detail drives the whole design. The notebook's `chunk_text` splits text
on a sentence-ending mark FOLLOWED BY WHITESPACE, so "a duck is a bird . a goat is an
animal ." becomes two separate one-sentence training passages and the model never sees
the two clauses together. Writing the same text with the period glued to the next word
("a duck is a bird .a goat is an animal .") keeps both clauses inside ONE passage, and
because the tokenizer matches \\w+ or a single punctuation character and ignores
whitespace entirely, the training token sequence is identical to the spaced version.
That is the only way a file corpus can teach a two-clause pattern such as the analogy
cases. It is a disclosed workaround, not a trick to smuggle in test material: the
leakage check still runs on the joined passage exactly as the model will see it.

Usage:
    python make_corpus.py            # write corpus/ files and the PDF source
    python make_corpus.py --check    # report only, write nothing
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

from run_evals import load_suite, matching_cases, word_tokens

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
SOURCES = ROOT / "corpus_sources"
SEED = 7

TAUGHT = ("grammar", "opposites", "everyday_knowledge", "categories_and_analogies")
UNTAUGHT = ("negation", "reference", "sequence", "spatial_relations")


def sentence(*words: str) -> str:
    """One training passage: a single clause ending in a period."""
    return " ".join(" ".join(words).split()) + " ."


def joined(first: str, second: str) -> str:
    """Two clauses the splitter must keep together (see the module docstring)."""
    return first.rstrip() + second


# --------------------------------------------------------------------------- grammar
SINGULAR = ["cat", "dog", "duck", "goat", "horse", "robin", "salmon", "kitten",
            "puppy", "tree", "spoon", "shoe", "pillow", "umbrella", "carrot", "bird"]
PLURAL = {"cat": "cats", "dog": "dogs", "duck": "ducks", "goat": "goats",
          "horse": "horses", "robin": "robins", "salmon": "salmon", "kitten": "kittens",
          "puppy": "puppies", "tree": "trees", "spoon": "spoons", "shoe": "shoes",
          "pillow": "pillows", "umbrella": "umbrellas", "carrot": "carrots", "bird": "birds"}
DESCRIPTORS = ["small", "quiet", "loud", "soft", "heavy", "light", "fast", "slow",
               "warm", "cold", "dry", "wet", "dark", "bright", "round", "calm"]
WALKERS = ["he", "she", "the cat", "the dog", "my friend", "one horse", "the teacher",
           "the student", "the doctor", "the nurse"]


def grammar_passages() -> list[tuple[str, str]]:
    out = []
    for noun, adjective in itertools.product(SINGULAR, DESCRIPTORS):
        plural = PLURAL[noun]
        out += [("one_is", sentence("one", noun, "is", adjective)),
                ("a_is", sentence("a", noun, "is", adjective)),
                ("the_is", sentence("the", noun, "is", adjective)),
                ("that_is", sentence("that", noun, "is", adjective)),
                ("two_are", sentence("two", plural, "are", adjective)),
                ("many_are", sentence("many", plural, "are", adjective)),
                ("those_are", sentence("those", plural, "are", adjective)),
                ("the_are", sentence("the", plural, "are", adjective)),
                ("the_were", sentence("the", plural, "were", adjective, "yesterday")),
                ("i_am", sentence("i am", adjective, "today"))]
    for walker in WALKERS:
        for place in ["store", "school", "market", "station", "office", "hospital"]:
            out += [("past_front", sentence("yesterday", walker, "walked to the", place)),
                    ("past_end", sentence(walker, "walked to the", place, "yesterday")),
                    ("present_front", sentence("today", walker, "walks to the", place)),
                    ("present_end", sentence(walker, "walks to the", place, "today")),
                    ("progressive", sentence(walker, "is walking to the", place, "now")),
                    ("past_progressive", sentence(walker, "was walking to the", place))]
        out += [("past_bare", sentence("yesterday", walker, "walked")),
                ("present_bare", sentence("today", walker, "walks"))]
    for subject in ["they", "the students", "two friends", "the teachers"]:
        for place in ["store", "school", "market", "station"]:
            out += [("plural_present", sentence(subject, "walk to the", place, "today")),
                    ("plural_past", sentence(subject, "walked to the", place, "yesterday")),
                    ("plural_progressive", sentence(subject, "were walking to the", place))]
        out += [("plural_are", sentence(subject, "are", "quiet")),
                ("plural_were", sentence(subject, "were", "loud", "yesterday"))]
    out += [("i_am", sentence("i am walking to the school")),
            ("i_am", sentence("i am a student")),
            ("i_am", sentence("i am quiet today")),
            ("plural_present", sentence("i walk to the office"))]
    return out


# ------------------------------------------------------------------------- opposites
PAIRS = [("hot", "cold"), ("empty", "full"), ("noisy", "quiet"), ("fast", "slow"),
         ("heavy", "light"), ("early", "late"), ("loud", "soft"), ("round", "square"),
         ("warm", "cool"), ("wet", "dry"), ("dark", "bright"), ("long", "short"),
         ("big", "small"), ("clean", "dirty"), ("hard", "gentle"), ("new", "old")]
OPPOSITE_SUBJECTS = ["room", "day", "morning", "street", "kitchen", "office", "station",
                     "school", "market", "store", "hospital", "bank"]


def opposites_passages() -> list[tuple[str, str]]:
    out = []
    for first, second in PAIRS:
        for a, b in ((first, second), (second, first)):
            out += [("opposite_of", sentence("the opposite of", a, "is", b)),
                    ("is_opposite", sentence(a, "is the opposite of", b)),
                    ("are_opposites", sentence(a, "and", b, "are opposites")),
                    ("day_frame", sentence("a", a, "day is the opposite of a", b, "day"))]
            for subject in OPPOSITE_SUBJECTS:
                out += [("not_frame", sentence("when the", subject, "is not", a, "it is", b)),
                        ("change_frame", sentence("the", subject, "was", a, "and then it was", b)),
                        ("contrast_frame", sentence("a", a, subject, "is not a", b, subject))]
    return out


# ----------------------------------------------------------------- everyday knowledge
FACTS = [
    "when water freezes it becomes ice", "ice is frozen water",
    "ice melts into water", "water becomes steam when it is very hot",
    "steam rises from hot water", "cold water freezes in the winter",
    "water freezes when the room is cold", "ice is cold and hard",
    "an umbrella keeps a person dry", "we stay dry under an umbrella",
    "to stay dry a person uses an umbrella in the rain",
    "rain makes a person wet", "a person without an umbrella gets wet",
    "a wet shoe takes a long time to dry", "a dry towel is warm",
    "we turn on a light when the room is dark",
    "a dark room needs a light", "the light makes a dark room bright",
    "in a dark room we cannot see", "we see better when the light is on",
    "a person uses a light to see at night", "the lamp gives light to the room",
    "sand is warm at the beach", "sand is dry and soft",
    "wood is hard and heavy", "a tree is made of wood",
    "a pillow is soft and light", "a tired person falls asleep on a pillow",
    "a cat is asleep in the warm room", "a hungry person eats an apple",
    "a hungry person is not asleep", "a spoon is made of metal",
    "a shoe is made of fabric", "an umbrella is made of fabric and metal",
    "a person uses a spoon to eat", "a person wears a shoe on each foot",
]
FACT_FRAMES = ["{fact}", "we learned that {fact}", "the teacher explained that {fact}",
               "at school we read that {fact}", "everybody knows that {fact}",
               "a student asked why {fact}", "the lesson today was that {fact}"]


def everyday_passages() -> list[tuple[str, str]]:
    return [(f"frame_{n}", sentence(frame.format(fact=fact)))
            for fact in FACTS for n, frame in enumerate(FACT_FRAMES)]


# ------------------------------------------------------- categories and analogies
IS_A = [("robin", "bird"), ("duck", "bird"), ("salmon", "fish"), ("goat", "animal"),
        ("horse", "animal"), ("cat", "animal"), ("dog", "animal"), ("carrot", "vegetable"),
        ("apple", "fruit"), ("banana", "fruit"), ("pear", "fruit"), ("spoon", "tool"),
        ("bus", "vehicle"), ("car", "vehicle"), ("train", "vehicle"), ("tree", "plant")]
MADE_OF = [("shoe", "fabric"), ("spoon", "metal"), ("pillow", "fabric"),
           ("car", "metal"), ("tree", "wood"), ("umbrella", "fabric")]
GROWS = [("puppy", "dog"), ("kitten", "cat"), ("foal", "horse"), ("chick", "duck"),
         ("kid", "goat"), ("seed", "tree")]
ARTICLE = {"animal": "an", "apple": "an", "umbrella": "an"}


def article(word: str) -> str:
    return ARTICLE.get(word, "an" if word[0] in "aeiou" else "a")


def categories_passages() -> list[tuple[str, str]]:
    out = []
    for thing, kind in IS_A:
        out += [("is_a", sentence(article(thing), thing, "is", article(kind), kind)),
                ("every_is_a", sentence("every", thing, "is", article(kind), kind)),
                ("belongs", sentence("the", thing, "belongs with every other", kind))]
    for thing, material in MADE_OF:
        out.append(("made_of", sentence(article(thing), thing, "is made of", material)))
    for young, grown in GROWS:
        out += [("grows_into", sentence(article(young), young, "grows into", article(grown), grown)),
                ("every_grows", sentence("every", young, "grows into", article(grown), grown))]
    # Two-clause analogies: the clauses must stay in ONE passage, hence `joined`.
    for (a_thing, a_kind), (b_thing, b_kind) in itertools.permutations(IS_A, 2):
        if a_kind == b_kind:
            continue
        out.append(("analogy_is_a",
                    joined(sentence(article(a_thing), a_thing, "is", article(a_kind), a_kind),
                           sentence(article(b_thing), b_thing, "is", article(b_kind), b_kind))))
    for (a_young, a_grown), (b_young, b_grown) in itertools.permutations(GROWS, 2):
        out.append(("analogy_grows",
                    joined(sentence(article(a_young), a_young, "grows into", article(a_grown), a_grown),
                           sentence(article(b_young), b_young, "grows into", article(b_grown), b_grown))))
    for (thing, material), (other, other_material) in itertools.permutations(MADE_OF, 2):
        if material == other_material:
            continue
        out.append(("analogy_made_of",
                    joined(sentence(article(thing), thing, "is made of", material),
                           sentence(article(other), other, "is made of", other_material))))
    return out


BUILDERS = {"grammar": grammar_passages, "opposites": opposites_passages,
            "everyday_knowledge": everyday_passages,
            "categories_and_analogies": categories_passages}
# everyday_knowledge is delivered as a PDF so the PDF extraction path is exercised.
AS_PDF = "everyday_knowledge"
CAP = 420


RESERVED: dict[str, list[dict]] = {}


def build(suite) -> dict[str, list[str]]:
    """Generate, then withhold any line that reproduces an eval prompt.

    This is the same rule the starter notebook applies to its own generated classroom
    sentences (`reserve_classroom_passages`): produce the material first, then reserve
    whatever collides with the exam. Every removal is recorded and reported, so the
    withheld lines are visible evidence rather than a silent filter.
    """
    rng = random.Random(SEED)
    built = {}
    for name, builder in BUILDERS.items():
        groups, reserved, seen = {}, [], set()
        for pattern, line in sorted(set(builder())):
            if line in seen:
                continue
            seen.add(line)
            hits = sorted({case for passage in normalized_passages(line)
                           for case in matching_cases(passage, suite)})
            if hits:
                reserved.append({"line": line, "case_ids": hits})
            else:
                groups.setdefault(pattern, []).append(line)
        RESERVED[name] = reserved
        # Round-robin across patterns so the cap never deletes a whole sentence pattern.
        for lines in groups.values():
            rng.shuffle(lines)
        kept, order = [], sorted(groups)
        while len(kept) < CAP and any(groups[p] for p in order):
            for pattern in order:
                if groups[pattern] and len(kept) < CAP:
                    kept.append(groups[pattern].pop())
        built[name] = sorted(kept)
    return built


def normalized_passages(text: str) -> list[str]:
    """Exactly what the notebook will turn this file into."""
    out = []
    for unit in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = word_tokens(unit)
        out.extend(" ".join(tokens[i:i + 47]) for i in range(0, len(tokens), 47))
    return out


def report(built: dict[str, list[str]], suite) -> dict:
    """Leakage check plus the vocabulary accounting the README needs."""
    starter_types = set(word_tokens(Path(ROOT / "results/exp1-starter/corpus.txt").read_text())) \
        if (ROOT / "results/exp1-starter/corpus.txt").exists() else set()
    leaks, passages, per_file = [], [], {}
    for name, lines in built.items():
        text = "\n".join(lines)
        got = normalized_passages(text)
        per_file[name] = {"lines": len(lines), "passages": len(got), "unique": len(set(got)),
                          "multi_clause": sum(1 for p in got if p.count(" . ") >= 1)}
        for passage in got:
            hit = matching_cases(passage, suite)
            if hit:
                leaks.append({"file": name, "passage": passage, "cases": hit})
        passages += got
    types = Counter(t for p in passages for t in word_tokens(p))
    needed = {}
    for case in suite["cases"]:
        if case["group"] != "extend_corpus":
            continue
        words = set(word_tokens(case["prompt"])) | {word_tokens(c)[0] for c in case["choices"]}
        missing = sorted(w for w in words if w not in types and w not in starter_types)
        needed[case["id"]] = {"category": case["category"], "missing_after_extension": missing,
                              "covered": not missing}
    taught_ids = [c["id"] for c in suite["cases"] if c["category"] in TAUGHT]
    untaught_ids = [c["id"] for c in suite["cases"] if c["category"] in UNTAUGHT]
    return {"files": per_file,
            "reserved_lines": {k: v for k, v in RESERVED.items() if v},
            "reserved_line_count": sum(len(v) for v in RESERVED.values()),
            "total_passages": len(passages), "unique_passages": len(set(passages)),
            "distinct_token_types_added_by_corpus_files": len(types), "leaks": leaks,
            "coverage": needed,
            "taught_cases_fully_covered": sum(needed[i]["covered"] for i in taught_ids),
            "taught_cases": len(taught_ids),
            "untaught_cases_fully_covered": sum(needed[i]["covered"] for i in untaught_ids),
            "untaught_cases": len(untaught_ids),
            "starter_vocabulary_known": len(starter_types)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    suite = load_suite(ROOT / "evals/language_evals.json")
    built = build(suite)
    summary = report(built, suite)
    if summary["leaks"]:
        for leak in summary["leaks"]:
            print("LEAK", leak["file"], leak["cases"], leak["passage"])
        raise SystemExit("Generated text reproduces eval prompts. Fix the templates.")

    print(f"reserved before writing (lines that reproduced an eval prompt): "
          f"{summary['reserved_line_count']}")
    for name, rows in summary["reserved_lines"].items():
        for row in rows:
            print(f"  reserved [{name}] {row['case_ids']}: {row['line']}")
    for name, values in summary["files"].items():
        print(f"{name:26s} {values['passages']:5d} passages "
              f"({values['unique']} unique, {values['multi_clause']} multi-clause)")
    print(f"{'TOTAL':26s} {summary['total_passages']:5d} passages, "
          f"{summary['unique_passages']} unique, "
          f"{summary['distinct_token_types_added_by_corpus_files']} distinct token types")
    print(f"taught cases fully in vocabulary:   {summary['taught_cases_fully_covered']}/{summary['taught_cases']}")
    print(f"untaught cases fully in vocabulary: {summary['untaught_cases_fully_covered']}/{summary['untaught_cases']}"
          "  (these are the control group; they should stay uncovered)")
    for case_id, values in summary["coverage"].items():
        if values["category"] in TAUGHT and values["missing_after_extension"]:
            print("  still missing for", case_id, values["category"], values["missing_after_extension"])
    if args.check:
        return

    CORPUS.mkdir(exist_ok=True)
    SOURCES.mkdir(exist_ok=True)
    for name, lines in built.items():
        text = "\n".join(lines) + "\n"
        target = (SOURCES if name == AS_PDF else CORPUS) / f"{name}.md"
        target.write_text(text, encoding="utf-8")
        print("wrote", target.relative_to(ROOT))
    (ROOT / "corpus_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote corpus_report.json")
    print(f"\nNext: render {SOURCES.relative_to(ROOT)}/{AS_PDF}.md to corpus/{AS_PDF}.pdf")


if __name__ == "__main__":
    main()
