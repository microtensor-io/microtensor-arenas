from __future__ import annotations

import re
from typing import Any

from arenas.compress.trajectory import Observed, Session, next_action

MAX_FILES = 12
MAX_ERRORS = 5
MAX_SYMBOLS = 8
BOUNDARY_BEFORE = r"(?<![\w.-])"
BOUNDARY_AFTER = r"(?![\w-])"


def normalise(text: str) -> str:
    cleaned = text.replace("\\", "/")
    for mark in ("`", '"', "'"):
        cleaned = cleaned.replace(mark, " ")
    return " ".join(cleaned.lower().split())


def mentions(text: str, alias: str) -> bool:
    pattern = BOUNDARY_BEFORE + re.escape(normalise(alias)) + BOUNDARY_AFTER
    return re.search(pattern, normalise(text)) is not None


def _file(path: str) -> dict[str, Any]:
    base = path.rsplit("/", 1)[-1]
    aliases = [path] if base == path else [path, base]
    return {"kind": "file", "value": path, "aliases": [normalise(a) for a in aliases]}


def _error(name: str) -> dict[str, Any]:
    short = name.rsplit(".", 1)[-1]
    aliases = [name] if short == name else [name, short]
    return {"kind": "error", "value": name, "aliases": [normalise(a) for a in aliases]}


def _symbol(name: str) -> dict[str, Any]:
    return {"kind": "symbol", "value": name, "aliases": [normalise(name)]}


def _recent(values: list[str], limit: int) -> list[str]:
    return values[-limit:]


def facts(seen: Observed) -> list[dict[str, Any]]:
    files: list[str] = []
    for path in [*seen.opened, *seen.created, *seen.edited]:
        if path not in files:
            files.append(path)
    keep = [p for p in files if p in seen.edited or p in seen.created]
    keep += [p for p in files if p not in keep]
    chosen_files = sorted(keep[:MAX_FILES], key=files.index)
    errors: list[str] = []
    for name, _ in seen.errors:
        if name in errors:
            errors.remove(name)
        errors.append(name)
    return [
        *[_file(p) for p in chosen_files],
        *[_error(e) for e in _recent(errors, MAX_ERRORS)],
        *[_symbol(s) for s in _recent(seen.symbols, MAX_SYMBOLS)],
    ]


def _probe(question: str, answer: str, *extra: str) -> dict[str, Any]:
    answers = [normalise(answer), *(normalise(e) for e in extra if e and e != answer)]
    return {"type": "exact", "question": question, "answers": answers}


def _word(question: str, answer: str) -> dict[str, Any]:
    return {"type": "word", "question": question, "answers": [answer]}


def probes(seen: Observed, session: Session | None = None, cut: int = 0) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if seen.edited:
        path = seen.edited[-1]
        found.append(
            _probe("Which file did the agent edit most recently?", path, path.rsplit("/", 1)[-1])
        )
    found.append(
        _word(
            "Has the agent changed any existing file in the repository yet? Answer yes or no.",
            "yes" if seen.edited else "no",
        )
    )
    if seen.tests:
        found.append(
            _word(
                "Did the agent's most recent test run pass or fail? Answer pass or fail.",
                "pass" if seen.tests[-1] else "fail",
            )
        )
    following = next_action(session, cut) if session is not None else None
    if following is not None:
        kind, path = following
        found.append(
            {
                "type": "next_step",
                "question": (
                    "What will the agent most likely do next? Answer with one of read, edit, "
                    "test, run, install or other, followed by the file it will touch, if any."
                ),
                "answers": {
                    "kind": kind,
                    "file": sorted({normalise(a) for a in (path, path.rsplit("/", 1)[-1]) if a}),
                },
            }
        )
    if seen.created:
        path = seen.created[-1]
        found.append(_probe("Which file did the agent create?", path, path.rsplit("/", 1)[-1]))
    if seen.errors:
        name = seen.errors[-1][0]
        found.append(
            _probe(
                "Which exception type appeared most recently in the output?",
                name,
                name.rsplit(".", 1)[-1],
            )
        )
    named = [(path, fn) for path, fn in seen.frames if fn != "<module>"]
    if named:
        found.append(_probe("In which function did the most recent traceback end?", named[-1][1]))
    return found
