from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from arenas.compress import facts, trajectory
from arenas.pool import TESTS, Pool

DATASET = "nebius/SWE-agent-trajectories"
TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SESSION_TOKENS = 27000
MIN_SESSION_TOKENS = 3000
MIN_FACTS = 4
MIN_PROBES = 2
TWIN_GAP = 6
ACCEPT = 0.25
CUTS_TRIED = 8
GOAL_CHOICES = 4
LETTERS = "abcd"
INSTRUCTION = (
    "You compact a coding agent's session so the agent can continue from your summary alone. "
    "The session is inside <session> tags. Reply with one JSON object and nothing else, with "
    "these fields: summary (the task, what was tried, what failed, where the work stands and the "
    "next step), files_touched (every file path the agent opened, created or edited), open_tasks "
    "(what is left to do), errors (each error or failing check, with its exact message), "
    "decisions (each change made or approach chosen, and why). Say which file the agent was "
    "working on last and whether its latest test run passed. Copy file paths, function names "
    "and error messages exactly as they appear. Use only facts from the session. Keep the "
    "reply within {budget} tokens."
)


def counter(name: str = TOKENIZER) -> Callable[[str], int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(name)
    return lambda text: len(tokenizer.encode(text).ids)


def _parquet_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=128, columns=["instance_id", "trajectory"]
        ):
            yield from batch.to_pylist()


def rows(source: str) -> Iterator[dict[str, Any]]:
    if source == "hf":
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(
            n
            for n in HfApi().list_repo_files(DATASET, repo_type="dataset")
            if n.startswith("data/train-") and n.endswith(".parquet")
        )
        for name in names:
            yield from _parquet_rows([Path(hf_hub_download(DATASET, name, repo_type="dataset"))])
        return
    root = Path(source)
    files = sorted(root.rglob("*")) if root.is_dir() else [root]
    for path in files:
        if path.suffix == ".parquet":
            yield from _parquet_rows([path])
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def _cuts(session: trajectory.Session, rng: random.Random, limit: int = CUTS_TRIED) -> list[int]:
    total = len(session.steps)
    found = [
        index
        for index in range(max(2, int(total * 0.35)), int(total * 0.9) + 1)
        if session.steps[index - 1].role == "tool"
    ]
    rng.shuffle(found)
    return found[:limit]


def make_item(
    session: trajectory.Session,
    cut: int,
    count: Callable[[str], int],
    item_id: str,
    test: str,
    twin_of: str = "",
) -> dict[str, Any] | None:
    seen = trajectory.observe(session, cut)
    found_facts = facts.facts(seen)
    found_probes = facts.probes(seen, session, cut)
    if len(found_facts) < MIN_FACTS or len(found_probes) < MIN_PROBES:
        return None
    user = trajectory.render(session, cut)
    session_tokens = count(user)
    if not MIN_SESSION_TOKENS <= session_tokens <= MAX_SESSION_TOKENS:
        return None
    budget = max(300, int(0.1 * (session_tokens + count(INSTRUCTION.format(budget=99999)))))
    system = INSTRUCTION.format(budget=budget)
    input_tokens = session_tokens + count(system)
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
            "instance": session.instance_id,
            "goal": trajectory.goal_of(session),
        },
        "tests": [test],
        "twin_of": twin_of,
    }


def _first(
    session: trajectory.Session,
    cuts: list[int],
    count: Callable[[str], int],
    item_id: str,
    test: str,
    twin_of: str = "",
) -> dict[str, Any] | None:
    for cut in cuts:
        item = make_item(session, cut, count, item_id, test, twin_of)
        if item is not None:
            return item
    return None


def generate(
    arena_id: str,
    seed: int,
    counts: Mapping[str, int],
    *,
    source: str = "hf",
    twins: tuple[str, ...] = ("slm",),
    count: Callable[[str], int] | None = None,
    per_session: int = 1,
) -> Pool:
    measure = count or counter()
    rng = random.Random(seed)
    pool = Pool(arena_id=arena_id, seed=seed)
    wanted = [(test, int(counts.get(test, 0))) for test in TESTS if counts.get(test)]
    made = {test: 0 for test, _ in wanted}
    used: set[str] = set()
    for row in rows(source):
        if all(made[test] >= n for test, n in wanted):
            break
        if source == "hf" and rng.random() > ACCEPT:
            continue
        session = trajectory.parse(row)
        if not session.instance_id or session.instance_id in used or len(session.steps) < 8:
            continue
        needed = [t for t, n in wanted if made[t] < n][: max(1, per_session)]
        cuts = _cuts(session, rng, CUTS_TRIED * max(1, per_session))
        taken: list[int] = []
        for test in needed:
            free = [c for c in cuts if all(abs(c - t) >= TWIN_GAP for t in taken)]
            stem = f"{arena_id}-s{seed}-{test}"
            item = _first(session, free, measure, f"{stem}-{made[test]:05d}", test)
            if item is None:
                continue
            cut = item["label"]["cut"]
            taken.append(cut)
            pool.items.append(item)
            if test in twins:
                others = [c for c in cuts if abs(c - cut) >= TWIN_GAP]
                twin = _first(
                    session,
                    others,
                    measure,
                    f"{stem}-twin-{made[test]:05d}",
                    test,
                    twin_of=item["item_id"],
                )
                if twin is not None:
                    pool.items.append(twin)
            made[test] += 1
        if taken:
            used.add(session.instance_id)
    add_goal_probes(pool, rng)
    return pool


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
                "question": f"Which issue is the agent working on? Answer A, B, C or D.\n{lines}",
                "answers": [LETTERS[options.index(goal)]],
            }
        )


def check(pool: Pool) -> list[str]:
    problems: list[str] = []
    for item in pool.items:
        label = item["label"]
        session = item["task"]["user"]
        if len(label["facts"]) < MIN_FACTS:
            problems.append(f"{item['item_id']}: {len(label['facts'])} facts")
        if len(label["probes"]) < MIN_PROBES:
            problems.append(f"{item['item_id']}: {len(label['probes'])} probes")
        for fact in label["facts"]:
            if not any(facts.mentions(session, alias) for alias in fact["aliases"]):
                problems.append(f"{item['item_id']}: fact {fact['value']} not in the session")
        for probe in label["probes"]:
            if probe.get("type", "exact") != "exact":
                continue
            if not any(facts.mentions(session, answer) for answer in probe["answers"]):
                problems.append(
                    f"{item['item_id']}: probe answer {probe['answers'][0]} not in the session"
                )
    return problems
