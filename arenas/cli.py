from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from arenas import definition, spec
from arenas.compress import build, fetch, hidden, synth
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
            "Build more dev items in the arena's exact format with",
            "",
            "```",
            "pip install -e '.[compress]'",
            f"mta dev --arena {arena_id} --count 1000 --seed 7 --out data/dev",
            "mta check data/dev",
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


def _sources(
    values: list[str] | None, count: int, tokens: tuple[int, int]
) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for value in values or ["synth"]:
        source, _, path = value.partition("=")
        if source == "synth" and not path:
            path = f"{count * 2}@{tokens[0]}-{tokens[1]}"
        if not path:
            raise ValueError(
                f"--source {value} needs a path, for example openhands=data/raw/openhands"
            )
        found.append((source, Path(path)))
    return found


def _dev(args: argparse.Namespace) -> int:
    if _task(args.arena) != COMPRESSION:
        raise ValueError(f"{args.arena} has no public dev generator")
    pool, report = build.generate(
        args.arena,
        args.seed,
        args.count,
        _sources(args.source, args.count, synth.DEV_TOKENS),
        min_tokens=args.min_tokens,
        only_resolved=args.resolved_only,
    )
    problems = build.check(pool)
    write(pool, Path(args.out))
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "families"}, indent=2))
    print(f"wrote {len(pool.items)} items to {args.out}, {len(problems)} problems")
    for line in problems[:20]:
        print(line)
    return 1 if problems else 0


def _hidden(args: argparse.Namespace) -> int:
    counts = dict(hidden.COUNTS)
    if args.epochs > 1:
        counts = {test: n * args.epochs for test, n in counts.items()}
    sessions = sum(counts.values())
    report = hidden.build_hidden(
        args.arena,
        args.seed or hidden.new_seed(),
        _sources(args.source, sessions, synth.HIDDEN_TOKENS),
        Path(args.out),
        counts=counts,
        stem=args.stem,
        since=date.fromisoformat(args.since) if args.since else None,
        ledger=Path(args.ledger) if args.ledger else None,
        min_tokens=args.min_tokens,
    )
    print(json.dumps(report, indent=2))
    short = {t: n for t, n in report["wanted"].items() if report["per_test"][t] < n}
    if short:
        print(f"short of {short}; add sessions with a larger synth count")
        return 1
    return 0


def _check(args: argparse.Namespace) -> int:
    pool = read(Path(args.pool))
    problems = build.check(pool)
    for line in problems[:50]:
        print(line)
    print(f"{len(pool.items)} items, {len(problems)} problems")
    return 1 if problems else 0


def _fetch(args: argparse.Namespace) -> int:
    written = fetch.fetch(args.source, Path(args.out), limit=args.limit, shards=args.shards)
    print(f"wrote {written} rows to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mta")
    sub = parser.add_subparsers(dest="command", required=True)

    rnd = sub.add_parser("render", help="write ARENA.md and the data note from each arena folder")
    rnd.add_argument("--arena", choices=definition.arena_ids(), default=None)
    rnd.add_argument("--check", action="store_true", help="fail when a rendered file is stale")
    rnd.set_defaults(handler=_render)

    dev = sub.add_parser("dev", help="build dev items in an arena's exact format")
    dev.add_argument("--arena", choices=definition.arena_ids(), required=True)
    dev.add_argument(
        "--count", type=int, default=1000, help="sessions; each gives an item and a twin"
    )
    dev.add_argument("--seed", type=int, default=1)
    dev.add_argument(
        "--source",
        action="append",
        help="synth (default), synth=COUNT@MIN-MAX, or openhands=, coderforge=, claude_code= a folder",
    )
    dev.add_argument("--min-tokens", type=int, default=build.DEV_MIN_TOKENS)
    dev.add_argument("--resolved-only", action="store_true")
    dev.add_argument("--out", required=True)
    dev.set_defaults(handler=_dev)

    hid = sub.add_parser("hidden", help="build hidden items from a secret seed")
    hid.add_argument("--arena", choices=definition.arena_ids(), required=True)
    hid.add_argument("--seed", default=None, help="secret seed; a fresh one is drawn when omitted")
    hid.add_argument("--epochs", type=int, default=1, help="epochs of items to build")
    hid.add_argument("--source", action="append")
    hid.add_argument("--stem", default="", help="item id prefix")
    hid.add_argument("--since", default=None, help="recorded sessions on or after YYYY-MM-DD only")
    hid.add_argument("--ledger", default=None, help="file of session ids already used")
    hid.add_argument("--min-tokens", type=int, default=build.HIDDEN_MIN_TOKENS)
    hid.add_argument("--out", required=True)
    hid.set_defaults(handler=_hidden)

    chk = sub.add_parser("check", help="check that every fact and probe answer is in its session")
    chk.add_argument("pool")
    chk.set_defaults(handler=_check)

    fet = sub.add_parser("fetch", help="download a public agent session corpus as jsonl")
    fet.add_argument("--source", choices=sorted(fetch.DATASETS), required=True)
    fet.add_argument("--out", required=True)
    fet.add_argument("--limit", type=int, default=None)
    fet.add_argument("--shards", type=int, default=None)
    fet.set_defaults(handler=_fetch)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
