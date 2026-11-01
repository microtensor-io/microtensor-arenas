from __future__ import annotations

import hashlib
import json
import random
import secrets
from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from arenas.compress import build
from arenas.compress.sources import Session
from arenas.pool import Pool, write

COUNTS = {"slm": 100, "harness": 100, "router": 60, "escalation": 100, "system": 100}


def new_seed() -> str:
    return secrets.token_hex(32)


def seed_digest(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _fresh(session: Session, since: date | None) -> bool:
    if since is None or session.source == "synth":
        return True
    try:
        stamp = datetime.fromisoformat(session.created_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return stamp.date() >= since


def read_ledger(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_ledger(path: Path, ids: Iterable[str], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for session_id in ids:
            handle.write(f"{session_id}\t{label}\n")


def build_hidden(
    arena_id: str,
    seed: str,
    sources: Iterable[tuple[str, Path]],
    out: Path,
    *,
    counts: dict[str, int] | None = None,
    stem: str = "",
    since: date | None = None,
    ledger: Path | None = None,
    min_tokens: int = build.HIDDEN_MIN_TOKENS,
    count: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    wanted = dict(counts or COUNTS)
    digest = seed_digest(seed)
    prefix = stem or f"{arena_id}-h{digest[:8]}"
    rng = random.Random(hashlib.sha256(f"{seed}:cuts".encode()).digest())
    used = read_ledger(ledger) if ledger else set()
    measure, tokenizer_name = (count, "custom") if count else build.counter()
    pool = Pool(arena_id=arena_id, seed=0)
    per_test = {test: 0 for test in wanted}
    sessions_used: list[str] = []
    skipped = {"stale": 0, "used": 0, "unusable": 0, "thin": 0, "no_cuts": 0}
    for session in build.sessions(sources, seed):
        if all(per_test[t] >= n for t, n in wanted.items()):
            break
        if not build.usable(session):
            skipped["unusable"] += 1
            continue
        if not _fresh(session, since):
            skipped["stale"] += 1
            continue
        if session.session_id in used:
            skipped["used"] += 1
            continue
        cuts = build.choose_cuts(session, rng)
        if cuts is None:
            skipped["no_cuts"] += 1
            continue
        test = rng.choice([t for t, n in wanted.items() if per_test[t] < n])
        item_id = f"{prefix}-{test}-{per_test[test]:04d}"
        half, late = cuts
        full = build.make_item(session, late, measure, item_id, [test], min_tokens=min_tokens)
        if full is None:
            skipped["thin"] += 1
            continue
        twin = build.make_item(
            session,
            half,
            measure,
            f"{item_id}-twin",
            [test],
            min_tokens=min_tokens // 2,
            twin_of=item_id,
        )
        pool.items.append(full)
        if twin is not None:
            pool.items.append(twin)
        per_test[test] += 1
        sessions_used.append(session.session_id)
    build.add_goal_probes(pool, rng)
    pool.items.sort(key=lambda i: i["item_id"])
    write(pool, out)
    report = {
        "arena_id": arena_id,
        "seed_sha256": digest,
        "stem": prefix,
        "wanted": wanted,
        "per_test": per_test,
        "items": len(pool.items),
        "twins": sum(1 for i in pool.items if i["twin_of"]),
        "tokenizer": tokenizer_name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_tokens": build.stats([i["label"]["input_tokens"] for i in pool.items]),
        "skipped": skipped,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ledger:
        append_ledger(ledger, sessions_used, prefix)
    return report
