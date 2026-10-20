from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

FENCE = re.compile(r"```([a-zA-Z]*)\n(.*?)```", re.DOTALL)
SHELLS = ("", "bash", "sh", "shell", "console")
FILE_HEADER = re.compile(r"\[File: (/[^\s\]]+) \(\d+ lines total\)\]")
FRAME = re.compile(r'File "([^"]+)", line \d+, in ([A-Za-z_]\w*)')
ERROR_LINE = re.compile(
    r"^\s*(?:E\d+\s+)?([A-Za-z_][\w.]*(?:Error|Exception))\b(?::\s*(.*))?$", re.MULTILINE
)
DEF = re.compile(
    r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(|^\s*class\s+([A-Za-z_]\w*)", re.MULTILINE
)
EDIT_COMMANDS = ("edit", "str_replace", "insert", "create", "sed -i", "cat >", "apply_patch")
AGENT_ROLES = ("ai", "assistant")
EXTENSIONS = (
    "py|pyi|js|ts|tsx|jsx|go|rs|java|rb|c|h|cc|cpp|hpp|toml|cfg|ini|yaml|yml|json|md|rst|txt|sh"
)
PATHLIKE = re.compile(rf"^(?:\./|/)?(?:[\w.@-]+/)*[\w@-][\w.@-]*\.(?:{EXTENSIONS})$")
READERS = frozenset(
    {"cat", "head", "tail", "nl", "less", "more", "grep", "sed", "wc", "awk", "python", "python3"}
)
REDIRECT = re.compile(r"(?:^|\s)(?:\d?>>?|tee(?:\s+-a)?)\s*['\"]?([^\s'\";|&]+)")
RETURN_CODE = re.compile(r"<returncode>(-?\d+)</returncode>")
TEST_COMMAND = re.compile(r"\b(?:pytest|py\.test|tox|nox|unittest|manage\.py test|make test)\b")
INSTALL_COMMAND = re.compile(
    r"\b(?:pip3? install|apt(?:-get)? install|conda install|uv pip install)\b"
)
READ_VERBS = frozenset(
    {
        "open",
        "goto",
        "scroll_down",
        "scroll_up",
        "search_file",
        "search_dir",
        "find_file",
        "find",
        "ls",
        "cat",
        "head",
        "tail",
        "nl",
        "grep",
        "sed",
        "less",
        "awk",
        "wc",
        "tree",
    }
)
EDIT_VERBS = frozenset({"edit", "str_replace", "insert", "create", "apply_patch"})
RUN_VERBS = frozenset({"python", "python3", "bash", "sh", "node"})
KINDS = ("read", "edit", "test", "run", "install", "other")
ISSUE_TAG = re.compile(r"<issue>\s*(.*?)\s*</issue>", re.DOTALL)
ISSUE_HEADER = re.compile(r"ISSUE:\s*\n?(.*)", re.DOTALL)


@dataclass(slots=True)
class Step:
    role: str
    text: str
    command: str = ""


@dataclass(slots=True)
class Session:
    instance_id: str
    repo: str
    steps: list[Step]
    root: str = ""

    def relative(self, path: str) -> str:
        found = path.replace("\\", "/")
        if self.root and found.startswith(self.root):
            found = found[len(self.root) :]
        return found.lstrip("/")

    def in_repo(self, path: str) -> bool:
        clean = path.replace("\\", "/")
        if "site-packages" in clean or "/lib/python" in clean or "/usr/" in clean:
            return False
        return not clean.startswith("/") or bool(self.root and clean.startswith(self.root))


@dataclass(slots=True)
class Observed:
    opened: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    frames: list[tuple[str, str]] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    tests: list[bool] = field(default_factory=list)


def _command(text: str) -> str:
    blocks = FENCE.findall(text or "")
    shells = [body for language, body in blocks if language.lower() in SHELLS]
    if shells:
        return str(shells[0]).strip()
    return str(blocks[-1][1]).strip() if blocks else ""


def _repo(instance_id: str) -> str:
    name = instance_id.split("__", 1)[-1]
    return re.sub(r"-\d+$", "", name)


def parse(row: Mapping[str, Any]) -> Session:
    instance_id = str(row.get("instance_id") or "")
    steps: list[Step] = []
    for raw in row.get("trajectory") or row.get("messages") or []:
        role = str(raw.get("role") or "")
        text = str(raw.get("text") or raw.get("content") or "")
        if role == "system" or not text:
            continue
        kind = "agent" if role in AGENT_ROLES else "tool"
        steps.append(Step(role=kind, text=text, command=_command(text) if kind == "agent" else ""))
    if steps:
        steps[0].role = "user"
    repo = str(row.get("repo") or "").split("/")[-1] or _repo(instance_id)
    session = Session(
        instance_id=instance_id, repo=repo, steps=steps, root=str(row.get("root") or "")
    )
    for step in steps if not session.root else []:
        header = FILE_HEADER.search(step.text)
        if header:
            parts = header.group(1).split("/")
            if len(parts) > 2:
                session.root = f"/{parts[1]}/"
                break
    return session


def _paths(tokens: list[str]) -> list[str]:
    cleaned = (token.strip("'\"") for token in tokens)
    return [token for token in cleaned if PATHLIKE.match(token)]


def shell_targets(command: str) -> tuple[list[str], list[str], list[str]]:
    lines: list[str] = []
    for line in command.splitlines():
        lines.append(line)
        if "<<" in line:
            break
    text = " ".join(lines)
    reads: list[str] = []
    edits: list[str] = []
    created: list[str] = []
    for segment in re.split(r"&&|\|\||;|\|", text):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if not tokens:
            continue
        verb = tokens[0]
        rest = [
            token
            for index, token in enumerate(tokens[1:], start=1)
            if not token.startswith(("-", ">", "<")) and tokens[index - 1] not in (">", ">>")
        ]
        if verb == "sed" and any(token.startswith("-i") for token in tokens[1:]):
            edits += _paths(rest[-1:])
        elif verb == "touch":
            created += _paths(rest)
        elif verb in READERS:
            reads += _paths(rest)
    created += [target for target in REDIRECT.findall(text) if PATHLIKE.match(target)]
    return reads, edits, created


def _succeeded(text: str) -> bool:
    found = RETURN_CODE.search(text)
    return found is None or found.group(1) == "0"


def kind_of(command: str) -> str:
    lines = command.strip().splitlines()
    first = lines[0] if lines else ""
    words = first.split()
    if not words:
        return "other"
    if TEST_COMMAND.search(first):
        return "test"
    if INSTALL_COMMAND.search(first):
        return "install"
    _, edits, created = shell_targets(command)
    if words[0] in EDIT_VERBS or edits or created:
        return "edit"
    if words[0] in READ_VERBS:
        return "read"
    if words[0] in RUN_VERBS:
        return "run"
    return "other"


def goal_of(session: Session) -> str:
    text = session.steps[0].text if session.steps else ""
    found = ISSUE_TAG.search(text) or ISSUE_HEADER.search(text)
    body = found.group(1) if found else text
    for line in body.splitlines():
        if line.strip():
            return " ".join(line.split())[:160]
    return ""


def next_action(session: Session, cut: int) -> tuple[str, str] | None:
    for step in session.steps[cut:]:
        if step.role != "agent" or not step.command:
            continue
        reads, edits, created = shell_targets(step.command)
        words = step.command.split()
        named = [words[1]] if words[0] in ("open", "create", "edit") and len(words) > 1 else []
        if words[0] == "edit":
            named = []
        targets = edits or created or named or reads
        return kind_of(step.command), session.relative(targets[0]) if targets else ""
    return None


def _passed(text: str) -> bool:
    found = RETURN_CODE.search(text)
    if found is not None:
        return found.group(1) == "0"
    lowered = text.lower()
    return " failed" not in lowered and "error" not in lowered


def _add(found: list[str], value: str) -> None:
    if value and value not in found:
        found.append(value)


def observe(session: Session, upto: int) -> Observed:
    seen = Observed()
    current = ""
    pending_edit = False
    pending: tuple[list[str], list[str], list[str]] = ([], [], [])
    testing = False
    for step in session.steps[:upto]:
        if step.role == "agent":
            command = step.command
            pending = shell_targets(command)
            testing = kind_of(command) == "test"
            first = command.splitlines()[0] if command else ""
            words = first.split()
            if words and words[0] == "open" and len(words) > 1:
                _add(seen.opened, session.relative(words[1]))
            elif words and words[0] == "create" and len(words) > 1:
                _add(seen.created, session.relative(words[1]))
            pending_edit = any(first.startswith(c) for c in EDIT_COMMANDS) and not first.startswith(
                "create"
            )
            if pending_edit:
                for name in DEF.findall(command):
                    _add(seen.symbols, name[0] or name[1])
            continue
        reads, edits, written = pending
        known = {*seen.opened, *seen.created, *seen.edited}
        for path in map(session.relative, written):
            _add(seen.edited if path in known else seen.created, path)
        if _succeeded(step.text):
            for path in map(session.relative, reads):
                _add(seen.opened, path)
            for path in map(session.relative, edits):
                _add(seen.edited, path)
        pending = ([], [], [])
        if testing:
            seen.tests.append(_passed(step.text))
        testing = False
        header = FILE_HEADER.search(step.text)
        if header:
            current = session.relative(header.group(1))
            _add(seen.opened, current)
        if pending_edit and current and "File updated" in step.text:
            _add(seen.edited, current)
        pending_edit = False
        for path, function in FRAME.findall(step.text):
            if session.in_repo(path):
                seen.frames.append((session.relative(path), function))
                if function != "<module>":
                    _add(seen.symbols, function)
        for name, message in ERROR_LINE.findall(step.text):
            seen.errors.append((name, (message or "").split("</")[0].strip()[:120]))
    return _canonical(seen)


def _canonical(seen: Observed) -> Observed:
    known = {*seen.opened, *seen.created, *seen.edited, *(p for p, _ in seen.frames)}

    def short(path: str) -> str:
        shorter = [q for q in known if q != path and path.endswith("/" + q)]
        return min(shorter, key=len) if shorter else path

    def unique(paths: list[str]) -> list[str]:
        found: list[str] = []
        for path in map(short, paths):
            _add(found, path)
        return found

    seen.opened = unique(seen.opened)
    seen.created = unique(seen.created)
    seen.edited = unique(seen.edited)
    seen.frames = [(short(p), fn) for p, fn in seen.frames]
    return seen


def render(session: Session, upto: int) -> str:
    parts: list[str] = []
    names = {"user": "user", "agent": "assistant", "tool": "tool"}
    for step in session.steps[:upto]:
        parts.append(f"[{names[step.role]}]\n{step.text.strip()}")
    return "<session>\n" + "\n\n".join(parts) + "\n</session>"
