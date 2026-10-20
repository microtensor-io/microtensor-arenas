from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TESTS = ("slm", "harness", "router", "escalation", "system")


@dataclass(slots=True)
class Pool:
    arena_id: str
    seed: int
    items: list[dict[str, Any]] = field(default_factory=list)
    blobs: dict[str, bytes] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for item in self.items:
            for test in item["tests"]:
                found[test] = found.get(test, 0) + 1
        return found


def write(pool: Pool, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "blobs").mkdir(exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item in pool.items:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    for name, body in pool.blobs.items():
        target = out / "blobs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "arena_id": pool.arena_id,
                "seed": pool.seed,
                "counts": pool.counts(),
                "items": len(pool.items),
                "blobs": len(pool.blobs),
                "families": sorted({i["family"] for i in pool.items}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read(out: Path) -> Pool:
    items = [
        json.loads(line)
        for line in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blobs: dict[str, bytes] = {}
    root = out / "blobs"
    for path in sorted(root.rglob("*")):
        if path.is_file():
            blobs[path.relative_to(root).as_posix()] = path.read_bytes()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return Pool(arena_id=summary["arena_id"], seed=int(summary["seed"]), items=items, blobs=blobs)
