from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from arenas.compress import canon, observe
from arenas.compress import facts as facts_module
from arenas.compress.sources import SOURCES, Session, parse, rows
from arenas.pool import Pool

TOKENIZER = "ibm-granite/granite-4.0-h-micro"
TOKENIZER_REVISION = "d5f01a3ea75f088947be3aae039f4ad52837dfde"
MAX_INPUT_TOKENS = 122_880
MAX_OUTPUT_TOKENS = 3072
MIN_OUTPUT_TOKENS = 1000
OUTPUT_SHARE = 0.05
DEV_MIN_TOKENS = 12_000
HIDDEN_MIN_TOKENS = 40_000
MIN_FACTS = 6
MIN_PROBES = 3
MIN_STEPS = 12
HALF = (0.45, 0.6)
LATE = (0.78, 0.94)
GOAL_CHOICES = 4
LETTERS = "abcd"


def counter(
    name: str = TOKENIZER, revision: str = TOKENIZER_REVISION
) -> tuple[Callable[[str], int], str]:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError(
            "token counts need the compress extra: pip install -e '.[compress]'"
        ) from exc
    tokenizer = Tokenizer.from_pretrained(name, revision=revision)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count, f"{name}@{revision[:8]}"


def budget_for(input_tokens: int) -> int:
    return int(min(MAX_OUTPUT_TOKENS, max(MIN_OUTPUT_TOKENS, OUTPUT_SHARE * input_tokens)))


def _tool_boundaries(session: Session) -> list[int]:
    return [
        i + 1
        for i, step in enumerate(session.steps)
        if step.role == "tool" and i + 1 < len(session.steps)
    ]


def choose_cuts(session: Session, rng: random.Random) -> tuple[int, int] | None:
    bounds = _tool_boundaries(session)
    if len(bounds) < 4:
        return None
    total = len(session.steps)
    late_options = [b for b in bounds if LATE[0] * total <= b <= LATE[1] * total] or bounds[-1:]
    late = rng.choice(late_options)
    middle = [b for b in bounds if HALF[0] * total <= b <= HALF[1] * total and b < late]
    if not middle:
        return None
    return rng.choice(middle), late


def make_item(
    session: Session,
    cut: int,
    count: Callable[[str], int],
    item_id: str,
    tests: list[str],
    *,
    min_tokens: int,
    twin_of: str = "",
    min_probes: int = MIN_PROBES,
) -> dict[str, Any] | None:
    found_facts, found_probes = observe.labels(session, cut)
    if len(found_facts) < MIN_FACTS or len(found_probes) < min_probes:
        return None
    user = canon.render(session, cut)
    session_tokens = count(user)
    if session_tokens < min_tokens:
        return None
    system_probe = canon.instruction(99999)
    input_tokens = session_tokens + count(system_probe)
    if input_tokens > MAX_INPUT_TOKENS:
        return None
    budget = budget_for(input_tokens)
    system = canon.instruction(budget)
    return {
        "item_id": item_id,
        "family": session.repo,
        "task": {"system": system, "user": user},
        "label": {
            "facts": found_facts,
            "probes": found_probes,
            "input_tokens": input_tokens,
            "budget_tokens": budget,
            "cut": cut,
            "steps": len(session.steps),
            "instance": session.session_id,
            "source": session.source,
            "resolved": session.resolved,
            "image": session.image,
            "goal": observe.goal_of(session),
        },
        "tests": list(tests),
        "twin_of": twin_of,
    }


def sessions(sources: Iterable[tuple[str, Path]], seed: str = "") -> Iterator[Session]:
    for source, path in sources:
        if source == "synth":
            from arenas.compress import synth

            count, tokens = synth.spec(str(path))
            for row in synth.rows(seed, count, tokens):
                yield parse(row, "synth")
            continue
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        for row in rows(path, source):
            yield parse(row, source)


def usable(session: Session) -> bool:
    return (
        bool(session.session_id)
        and len(session.steps) >= MIN_STEPS
        and any(step.role == "agent" and step.calls for step in session.steps)
    )


def generate(
    arena_id: str,
    seed: int,
    count_wanted: int,
    sources: Iterable[tuple[str, Path]],
    *,
    tests: tuple[str, ...] = ("slm",),
    min_tokens: int = DEV_MIN_TOKENS,
    exclude_families: set[str] | None = None,
    only_resolved: bool = False,
    count: Callable[[str], int] | None = None,
    stem: str = "",
) -> tuple[Pool, dict[str, Any]]:
    measure, tokenizer_name = (count, "custom") if count else counter()
    rng = random.Random(seed)
    pool = Pool(arena_id=arena_id, seed=seed)
    excluded = exclude_families or set()
    made = 0
    seen_ids: set[str] = set()
    skipped = {"unusable": 0, "family": 0, "unresolved": 0, "short_or_thin": 0, "no_cuts": 0}
    prefix = stem or f"{arena_id}-s{seed}"
    for session in sessions(sources, str(seed)):
        if made >= count_wanted:
            break
        if not usable(session) or session.session_id in seen_ids:
            skipped["unusable"] += 1
            continue
        if session.repo in excluded:
            skipped["family"] += 1
            continue
        if only_resolved and session.resolved is not True:
            skipped["unresolved"] += 1
            continue
        cuts = choose_cuts(session, rng)
        if cuts is None:
            skipped["no_cuts"] += 1
            continue
        half, late = cuts
        item_id = f"{prefix}-{made:05d}"
        full = make_item(session, late, measure, item_id, list(tests), min_tokens=min_tokens)
        if full is None:
            skipped["short_or_thin"] += 1
            continue
        twin = make_item(
            session,
            half,
            measure,
            f"{item_id}-twin",
            list(tests),
            min_tokens=min_tokens // 2,
            twin_of=item_id,
        )
        pool.items.append(full)
        if twin is not None:
            pool.items.append(twin)
        seen_ids.add(session.session_id)
        made += 1
    add_goal_probes(pool, rng)
    report = {
        "tokenizer": tokenizer_name,
        "sessions": made,
        "items": len(pool.items),
        "skipped": skipped,
        "families": sorted({i["family"] for i in pool.items}),
        "input_tokens": stats([i["label"]["input_tokens"] for i in pool.items]),
    }
    return pool, report


def stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"min": ordered[0], "p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1]}


def add_goal_probes(pool: Pool, rng: random.Random) -> None:
    goals = {item["family"]: item["label"].get("goal", "") for item in pool.items}
    for item in pool.items:
        goal = item["label"].pop("goal", "")
        others = sorted({g for f, g in goals.items() if g and f != item["family"] and g != goal})
        if not goal or len(others) < GOAL_CHOICES - 1:
            continue
        options = [goal, *rng.sample(others, GOAL_CHOICES - 1)]
        rng.shuffle(options)
        lines = "\n".join(f"{LETTERS[i].upper()}) {text}" for i, text in enumerate(options))
        item["label"]["probes"].append(
            {
                "type": "choice",
                "question": f"Which task is the agent working on? Answer A, B, C or D.\n{lines}",
                "answers": [LETTERS[options.index(goal)]],
            }
        )


def _attached_problems(item_id: str, session: str, attached: list[dict[str, Any]]) -> list[str]:
    flat = " ".join(session.split())
    problems: list[str] = []
    for probe in attached:
        keys = [" ".join(str(key).split()) for key in probe.get("keys") or []]
        missing = [key for key in keys if key not in flat]
        if probe.get("in_session", True) and missing:
            problems.append(f"{item_id}: attached probe {probe.get('id')} is not in the session")
        elif not probe.get("in_session", True) and keys and not missing:
            problems.append(f"{item_id}: attached probe {probe.get('id')} is in the session")
    return problems


def check(pool: Pool) -> list[str]:
    problems: list[str] = []
    for item in pool.items:
        label = item["label"]
        session = item["task"]["user"]
        problems += _attached_problems(item["item_id"], session, label.get("attached") or [])
        if len(label["facts"]) < MIN_FACTS:
            problems.append(f"{item['item_id']}: {len(label['facts'])} facts")
        if len(label["probes"]) < MIN_PROBES:
            problems.append(f"{item['item_id']}: {len(label['probes'])} probes")
        for fact in label["facts"]:
            if not any(facts_module.mentions(session, alias) for alias in fact["aliases"]):
                problems.append(f"{item['item_id']}: fact {fact['value']} not in the session")
        for probe in label["probes"]:
            if probe.get("type", "exact") != "exact":
                continue
            if not any(facts_module.mentions(session, answer) for answer in probe["answers"]):
                problems.append(
                    f"{item['item_id']}: probe answer {probe['answers'][0]} not in the session"
                )
        if label["input_tokens"] > MAX_INPUT_TOKENS:
            problems.append(f"{item['item_id']}: {label['input_tokens']} tokens over the envelope")
    return problems
