from __future__ import annotations

import re
from typing import Any

from arenas.compress import facts
from arenas.compress.facts import Observed
from arenas.compress.sources import Session

FRAME = re.compile(r'File "([^"]+)", line \d+, in ([A-Za-z_]\w*)')
PYTEST_FRAME = re.compile(r"^([\w./-]+\.py):\d+: in ([A-Za-z_]\w*)", re.MULTILINE)
ERROR_LINE = re.compile(
    r"^\s*(?:E\d*\s+)?([A-Za-z_][\w.]*(?:Error|Exception|Panic|Fault))\b(?::\s*(.*))?$",
    re.MULTILINE,
)
DEF = re.compile(
    r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(|^\s*class\s+([A-Za-z_]\w*)", re.MULTILINE
)
PYTEST_TAIL = re.compile(
    r"^([\w./-]+\.py):\d+: ([A-Za-z_]\w*(?:Error|Exception))\s*$", re.MULTILINE
)
PASSED = re.compile(r"\b(\d+) passed\b")
FAILED = re.compile(r"\b(\d+) (?:failed|error)s?\b")
MISSING = ("No such file or directory", "does not exist", "not found", "cannot access")


def _add(found: list[str], value: str) -> None:
    if value and value not in found:
        found.append(value)


def _in_repo(session: Session, path: str) -> bool:
    clean = path.replace("\\", "/")
    if "site-packages" in clean or "/lib/python" in clean or clean.startswith("/usr/"):
        return False
    return (
        not clean.startswith("/")
        or bool(session.root and clean.startswith(session.root))
        or clean.startswith(("/workspace/", "/testbed/"))
    )


def _test_passed(text: str, ok: bool | None) -> bool:
    if FAILED.search(text) and not re.search(r"\b0 (?:failed|errors?)\b", text):
        return False
    if PASSED.search(text):
        return True
    if ok is not None:
        return ok
    lowered = text.lower()
    return " failed" not in lowered and "error" not in lowered


def observe(session: Session, upto: int) -> Observed:
    seen = Observed()
    pending: list[tuple[str, str, str]] = []
    known: set[str] = set()
    for step in session.steps[:upto]:
        if step.role == "agent":
            pending = []
            for call in step.calls:
                path = session.relative(call.path) if call.path else ""
                pending.append((call.kind, path, call.name))
                if call.kind == "edit":
                    for name in DEF.findall(call.args):
                        _add(seen.symbols, name[0] or name[1])
            continue
        if step.role != "agent" and pending:
            kind, path, name = pending.pop(0)
            text = step.text
            if path and _in_repo(session, path) or (path and not path.startswith("/")):
                if kind == "read" and not any(m in text for m in MISSING):
                    _add(seen.opened, path)
                elif kind == "edit" and step.ok is not False:
                    if (
                        name in ("Write", "create")
                        or (name == "str_replace_editor" and "created" in text.lower())
                        or path not in known
                    ):
                        target = (
                            seen.created
                            if path not in known
                            and (name in ("Write", "create") or "created" in text.lower())
                            else seen.edited
                        )
                        _add(target, path)
                    else:
                        _add(seen.edited, path)
                known.add(path)
            if kind == "test":
                seen.tests.append(_test_passed(text, step.ok))
        for path, function in [*FRAME.findall(step.text), *PYTEST_FRAME.findall(step.text)]:
            if _in_repo(session, path):
                seen.frames.append((session.relative(path), function))
                if function != "<module>":
                    _add(seen.symbols, function)
        for name, message in ERROR_LINE.findall(step.text):
            seen.errors.append((name, (message or "").strip()[:120]))
        for path, name in PYTEST_TAIL.findall(step.text):
            seen.errors.append((name, f"at {path}"))
            if _in_repo(session, path):
                seen.frames.append((session.relative(path), "<module>"))
    return seen


def next_action(session: Session, cut: int) -> tuple[str, str] | None:
    for step in session.steps[cut:]:
        if step.role != "agent" or not step.calls:
            continue
        call = step.calls[0]
        return call.kind, session.relative(call.path) if call.path else ""
    return None


def goal_of(session: Session) -> str:
    text = session.steps[0].text if session.steps else ""
    found = re.search(
        r"<(?:issue|pr_description|task)>\s*(.*?)\s*</(?:issue|pr_description|task)>",
        text,
        re.DOTALL,
    )
    body = found.group(1) if found else text
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if len(line) > 12:
            return " ".join(line.split())[:160]
    return ""


def labels(session: Session, cut: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = observe(session, cut)
    found_facts = facts.facts(seen)
    found_probes = facts.probes(seen)
    following = next_action(session, cut)
    if following is not None:
        kind, path = following
        found_probes.append(
            {
                "type": "next_step",
                "question": (
                    "What will the agent most likely do next? Answer with one of read, edit, run, "
                    "test, search, install or other, followed by the file it will touch, if any."
                ),
                "answers": {
                    "kind": kind,
                    "file": sorted(
                        {facts.normalise(a) for a in (path, path.rsplit("/", 1)[-1]) if a}
                    ),
                },
            }
        )
    return found_facts, found_probes
