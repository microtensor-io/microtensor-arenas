from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arenas import definition, spec
from arenas.compress import generate as compress
from arenas.pool import read, write

COMPRESSION = "context_compression"


def _task(arena_id: str) -> str:
    return str(definition.load(arena_id)["task"])


def _data_note(arena_id: str, dataset: dict[str, str] | None) -> str:
    if not dataset:
        return "# Dataset\n\nNot attached yet.\n"
    lines = [
        "# Dataset",
        "",
        f"[{dataset['name']}]({dataset['url']})",
        "",
        f"Licence: {dataset.get('licence') or 'see the dataset card'}",
    ]
    if dataset.get("about"):
        lines += ["", " ".join(str(dataset["about"]).split())]
    if _task(arena_id) == COMPRESSION:
        lines += [
            "",
            "Build dev items in the arena's exact format from it with",
            "",
            "```",
            f"mta dev --arena {arena_id} --count 200 --out data/dev",
            "```",
        ]
    return "\n".join(lines) + "\n"


def _render(args: argparse.Namespace) -> int:
    stale: list[str] = []
    for arena_id in [args.arena] if args.arena else definition.arena_ids():
        arena = definition.load(arena_id)
        folder = definition.folder(arena_id)
        files = {
            folder / "ARENA.md": spec.markdown(spec.build(arena)),
            folder / "data" / "README.md": _data_note(arena_id, arena["slm"].get("dataset")),
        }
        for path, text in files.items():
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current == text:
                continue
            if args.check:
                stale.append(str(path.relative_to(definition.ROOT)))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(definition.ROOT)}")
    if stale:
        print("out of date: " + ", ".join(stale))
        return 1
    return 0


def _dev(args: argparse.Namespace) -> int:
    if _task(args.arena) != COMPRESSION:
        raise ValueError(f"{args.arena} has no public dev generator")
    pool = compress.generate(
        args.arena, args.seed, {"slm": args.count}, source=args.source, twins=("slm",)
    )
    problems = compress.check(pool)
    write(pool, Path(args.out))
    print(json.dumps(pool.counts(), indent=2))
    print(f"wrote {len(pool.items)} items to {args.out}, {len(problems)} problems")
    return 1 if problems else 0


def _check(args: argparse.Namespace) -> int:
    pool = read(Path(args.pool))
    problems = compress.check(pool)
    for line in problems[:50]:
        print(line)
    print(f"{len(pool.items)} items, {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mta")
    sub = parser.add_subparsers(dest="command", required=True)

    rnd = sub.add_parser("render", help="write ARENA.md and the data note from each arena folder")
    rnd.add_argument("--arena", choices=definition.arena_ids(), default=None)
    rnd.add_argument("--check", action="store_true", help="fail when a rendered file is stale")
    rnd.set_defaults(handler=_render)

    dev = sub.add_parser("dev", help="build dev items in an arena's exact format")
    dev.add_argument("--arena", choices=definition.arena_ids(), required=True)
    dev.add_argument("--count", type=int, default=200)
    dev.add_argument("--seed", type=int, default=1)
    dev.add_argument("--source", default="hf", help="hf, or a folder of trajectories")
    dev.add_argument("--out", required=True)
    dev.set_defaults(handler=_dev)

    chk = sub.add_parser("check", help="check that every fact and probe answer is in its session")
    chk.add_argument("pool")
    chk.set_defaults(handler=_check)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
