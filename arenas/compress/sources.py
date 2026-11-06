from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCES = ("openhands", "coderforge", "claude_code", "synth", "pi")
PI_CONTEXT = ("message", "custom_message", "branch_summary", "compaction")
KINDS = ("read", "edit", "run", "test", "search", "install", "other")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TEST_COMMAND = re.compile(
    r"\b(?:pytest|py\.test|tox|nox|unittest|manage\.py test|make test|npm test|cargo test|go test)\b"
)
INSTALL_COMMAND = re.compile(
    r"\b(?:pip3? install|apt(?:-get)? install|conda install|uv pip install|npm install|cargo add)\b"
)
SEARCH_COMMAND = re.compile(
    r"^\s*(?:grep|rg|ag|find|fd|ls|tree|git grep|git log|git status|git diff)\b"
)
READ_COMMAND = re.compile(r"^\s*(?:cat|head|tail|nl|less|more|sed -n|awk|wc|bat)\b")
RETURN_CODE = re.compile(
    r"(?:\[Command finished with exit code (-?\d+)\]|<returncode>(-?\d+)</returncode>|exit code[:=]\s*(-?\d+))"
)
PATH_IN_TEXT = re.compile(
    r"(?:^|\s|['\"(])((?:/|\./)?(?:[\w.@-]+/)+[\w@-][\w.@-]*\.[A-Za-z0-9]{1,8})"
)

TOOL_KINDS = {
    "str_replace_editor": None,
    "execute_bash": None,
    "execute_ipython_cell": "run",
    "browser": "other",
    "web_read": "other",
    "think": "other",
    "finish": "other",
    "Read": "read",
    "Edit": "edit",
    "MultiEdit": "edit",
    "Write": "edit",
    "NotebookEdit": "edit",
    "Bash": None,
    "Grep": "search",
    "Glob": "search",
    "LS": "search",
    "WebFetch": "other",
    "WebSearch": "other",
    "Task": "other",
    "TodoWrite": "other",
    "shell": None,
    "apply_patch": "edit",
    "bash": None,
    "edit": "edit",
    "create": "edit",
    "open": "read",
    "search_file": "search",
    "search_dir": "search",
    "find_file": "search",
    "read": "read",
    "write": "edit",
    "grep": "search",
    "find": "search",
    "ls": "search",
}


@dataclass(slots=True)
class ToolCall:
    name: str
    kind: str
    path: str = ""
    args: str = ""
    call_id: str = ""


@dataclass(slots=True)
class Step:
    role: str
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    ok: bool | None = None
    lines: int = 0
    call_id: str = ""


@dataclass(slots=True)
class Session:
    session_id: str
    repo: str
    source: str
    steps: list[Step]
    resolved: bool | None = None
    image: str = ""
    licence: str = ""
    created_at: str = ""
    root: str = ""

    def relative(self, path: str) -> str:
        found = path.replace("\\", "/")
        if self.root and found.startswith(self.root):
            found = found[len(self.root) :]
        for prefix in ("/workspace/", "/testbed/", "/repo/", "/app/"):
            if found.startswith(prefix):
                found = found[len(prefix) :]
                found = (
                    found.split("/", 1)[1]
                    if self.source == "openhands" and "/" in found and not self.root
                    else found
                )
                break
        return found.lstrip("./").lstrip("/")


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


def _shell_kind(command: str) -> str:
    first = command.strip().splitlines()[0] if command.strip() else ""
    if TEST_COMMAND.search(first):
        return "test"
    if INSTALL_COMMAND.search(first):
        return "install"
    if re.search(
        r"(?:^|\s)(?:sed -i|tee|>\s*\S|>>\s*\S|cat <<|cat >|apply_patch|git apply|patch )", first
    ):
        return "edit"
    if SEARCH_COMMAND.match(first):
        return "search"
    if READ_COMMAND.match(first):
        return "read"
    return "run"


def _shell_path(command: str) -> str:
    found = PATH_IN_TEXT.findall(command.splitlines()[0] if command else "")
    return found[0] if found else ""


def classify(name: str, arguments: Mapping[str, Any]) -> tuple[str, str, str]:
    kind = TOOL_KINDS.get(name, "other")
    path = str(
        arguments.get("path") or arguments.get("file_path") or arguments.get("filename") or ""
    )
    if name == "str_replace_editor":
        sub = str(arguments.get("command") or "")
        kind = {
            "view": "read",
            "create": "edit",
            "str_replace": "edit",
            "insert": "edit",
            "undo_edit": "edit",
        }.get(sub, "other")
        short = sub
    elif name in ("execute_bash", "Bash", "shell", "bash"):
        command = arguments.get("command") or arguments.get("cmd") or ""
        if isinstance(command, list):
            command = " ".join(str(c) for c in command)
        command = str(command)
        kind = _shell_kind(command)
        path = path or _shell_path(command)
        short = command.strip().splitlines()[0][:200] if command.strip() else ""
    elif name in ("Grep", "Glob", "search_file", "search_dir", "find_file", "grep", "find"):
        short = str(arguments.get("pattern") or arguments.get("query") or "")[:120]
    elif name == "apply_patch":
        patch = str(arguments.get("input") or arguments.get("patch") or "")
        files = re.findall(r"^\*\*\* (?:Update|Add|Delete) File: (\S+)", patch, re.MULTILINE)
        path = files[0] if files else path
        short = ", ".join(files[:4])
    else:
        short = ""
    if kind is None:
        kind = "other"
    return kind, path, short


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _result_meta(text: str) -> tuple[bool | None, int]:
    found = RETURN_CODE.search(text)
    lines = text.count("\n") + 1 if text else 0
    if found:
        code = next(g for g in found.groups() if g is not None)
        return code == "0", lines
    lowered = text.lower()
    if "traceback (most recent call last)" in lowered or lowered.startswith("error"):
        return False, lines
    return None, lines


def _openhands_steps(messages: list[Mapping[str, Any]]) -> list[Step]:
    steps: list[Step] = []
    for raw in messages:
        role = str(raw.get("role") or "")
        text = ANSI.sub("", _text_of(raw.get("content")))
        if role == "system":
            continue
        if role == "user":
            steps.append(Step(role="user", text=text))
        elif role == "assistant":
            calls = []
            for call in raw.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                arguments = _loads(function.get("arguments") or call.get("arguments"))
                if not isinstance(arguments, dict):
                    arguments = {}
                kind, path, short = classify(name, arguments)
                calls.append(
                    ToolCall(
                        name=name,
                        kind=kind,
                        path=path,
                        args=short,
                        call_id=str(call.get("id") or ""),
                    )
                )
            steps.append(Step(role="agent", text=text, calls=calls))
        elif role == "tool":
            ok, lines = _result_meta(text)
            steps.append(
                Step(
                    role="tool",
                    text=text,
                    ok=ok,
                    lines=lines,
                    call_id=str(raw.get("tool_call_id") or ""),
                )
            )
    return steps


def _claude_steps(messages: list[Mapping[str, Any]]) -> list[Step]:
    steps: list[Step] = []
    for raw in messages:
        role = str(raw.get("role") or "")
        content = raw.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "assistant":
            text_parts, calls = [], []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "tool_use":
                    name = str(block.get("name") or "")
                    arguments = block.get("input") or {}
                    kind, path, short = classify(
                        name, arguments if isinstance(arguments, dict) else {}
                    )
                    calls.append(
                        ToolCall(
                            name=name,
                            kind=kind,
                            path=path,
                            args=short,
                            call_id=str(block.get("id") or ""),
                        )
                    )
            steps.append(Step(role="agent", text=ANSI.sub("", "\n".join(text_parts)), calls=calls))
        elif role == "user":
            texts, results = [], []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    body = ANSI.sub("", _text_of(block.get("content")))
                    ok = (
                        not bool(block.get("is_error"))
                        if "is_error" in block
                        else _result_meta(body)[0]
                    )
                    results.append(
                        Step(
                            role="tool",
                            text=body,
                            ok=ok,
                            lines=body.count("\n") + 1 if body else 0,
                            call_id=str(block.get("tool_use_id") or ""),
                        )
                    )
                elif block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
            steps.extend(results)
            if any(t.strip() for t in texts):
                steps.append(Step(role="user", text=ANSI.sub("", "\n".join(texts))))
    return steps


def _pi_message(raw: Mapping[str, Any]) -> list[Step]:
    role = str(raw.get("role") or "")
    content = raw.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    if role == "user":
        return [Step(role="user", text=ANSI.sub("", _text_of(content)))]
    if role == "assistant":
        text_parts, calls = [], []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "toolCall":
                name = str(block.get("name") or "")
                arguments = _loads(block.get("arguments"))
                kind, path, short = classify(name, arguments if isinstance(arguments, dict) else {})
                calls.append(
                    ToolCall(
                        name=name,
                        kind=kind,
                        path=path,
                        args=short,
                        call_id=str(block.get("id") or ""),
                    )
                )
        return [Step(role="agent", text=ANSI.sub("", "\n".join(text_parts)), calls=calls)]
    if role == "toolResult":
        body = ANSI.sub("", _text_of(content))
        ok = not bool(raw.get("isError")) if "isError" in raw else _result_meta(body)[0]
        return [
            Step(
                role="tool",
                text=body,
                ok=ok,
                lines=body.count("\n") + 1 if body else 0,
                call_id=str(raw.get("toolCallId") or ""),
            )
        ]
    if role == "bashExecution":
        if raw.get("excludeFromContext"):
            return []
        shown = f"$ {raw.get('command') or ''}\n{raw.get('output') or ''}"
        return [Step(role="user", text=ANSI.sub("", shown).strip())]
    if role in ("compactionSummary", "branchSummary"):
        return [Step(role="user", text=str(raw.get("summary") or ""))]
    return []


def _pi_steps(entries: list[Mapping[str, Any]]) -> list[Step]:
    steps: list[Step] = []
    starts: dict[str, int] = {}
    for entry in entries:
        kind = entry.get("type")
        starts[str(entry.get("id") or "")] = len(steps)
        if kind == "message":
            steps.extend(_pi_message(entry.get("message") or {}))
        elif kind == "custom_message":
            text = ANSI.sub("", _text_of(entry.get("content")))
            if text.strip():
                steps.append(Step(role="user", text=text))
        elif kind == "branch_summary":
            steps.append(Step(role="user", text=str(entry.get("summary") or "")))
        elif kind == "compaction":
            kept = starts.get(str(entry.get("firstKeptEntryId") or ""), len(steps))
            steps = [Step(role="user", text=str(entry.get("summary") or "")), *steps[kept:]]
            starts = {}
    return steps


def pi_row(path: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                found = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(found, dict):
                continue
            if found.get("type") == "session" and not header:
                header = found
            elif "id" in found:
                entries.append(found)
    by_id = {entry["id"]: entry for entry in entries}
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = entries[-1] if entries else None
    while current is not None and current["id"] not in seen:
        seen.add(current["id"])
        chain.append(current)
        current = by_id.get(current.get("parentId"))
    chain.reverse()
    return {
        "session_id": str(header.get("id") or path.stem),
        "root": str(header.get("cwd") or ""),
        "recorded_at": str(header.get("timestamp") or ""),
        "entries": [entry for entry in chain if entry.get("type") in PI_CONTEXT],
    }


def _repo_of(instance_id: str) -> str:
    name = instance_id.split("__", 1)[-1]
    return re.sub(r"[-_]\d+(?:_run\d+)?$", "", name)


def parse(row: Mapping[str, Any], source: str) -> Session:
    if source == "openhands":
        messages = _loads(row.get("trajectory")) or []
        instance = str(row.get("instance_id") or "")
        session = Session(
            session_id=instance,
            repo=str(row.get("repo") or "").split("/")[-1] or _repo_of(instance),
            source=source,
            steps=_openhands_steps(messages),
            resolved=row.get("resolved") if isinstance(row.get("resolved"), bool) else None,
            image=str(row.get("image") or ""),
            licence=str(row.get("license") or ""),
            created_at=str(row.get("created_at") or ""),
        )
    elif source == "coderforge":
        messages = _loads(row.get("messages")) or []
        instance = str(row.get("trajectory_id") or "")
        reward = row.get("reward")
        session = Session(
            session_id=instance,
            repo=_repo_of(instance.split("_run")[0]),
            source=source,
            steps=_openhands_steps(messages),
            resolved=(float(reward) >= 1.0) if isinstance(reward, (int, float)) else None,
            image=str(row.get("image") or ""),
            licence=str(row.get("license") or ""),
            created_at=str(row.get("created_at") or ""),
        )
    elif source in ("claude_code", "synth"):
        messages = _loads(row.get("messages")) or []
        session = Session(
            session_id=str(row.get("session_id") or ""),
            repo=str(row.get("repo") or "").split("/")[-1],
            source=source,
            steps=_claude_steps(messages),
            resolved=row.get("resolved") if isinstance(row.get("resolved"), bool) else None,
            image=str(row.get("image") or ""),
            licence=str(row.get("license") or ""),
            created_at=str(row.get("recorded_at") or row.get("created_at") or ""),
            root=str(row.get("root") or ""),
        )
    elif source == "pi":
        entries = _loads(row.get("entries")) or []
        root = str(row.get("root") or "").replace("\\", "/")
        session = Session(
            session_id=str(row.get("session_id") or ""),
            repo=str(row.get("repo") or "").split("/")[-1] or root.rstrip("/").split("/")[-1],
            source=source,
            steps=_pi_steps(entries),
            resolved=row.get("resolved") if isinstance(row.get("resolved"), bool) else None,
            image=str(row.get("image") or ""),
            licence=str(row.get("license") or ""),
            created_at=str(row.get("recorded_at") or row.get("created_at") or ""),
            root=root.rstrip("/") + "/" if root else "",
        )
    else:
        raise ValueError(f"unknown source {source!r}; known: {', '.join(SOURCES)}")
    if session.steps and session.steps[0].role != "user":
        session.steps.insert(0, Step(role="user", text=""))
    _resolve_roots(session)
    return session


def _resolve_roots(session: Session) -> None:
    if session.root:
        return
    for step in session.steps:
        for call in step.calls:
            if call.path.startswith("/workspace/"):
                parts = call.path.split("/")
                if len(parts) > 3:
                    session.root = "/".join(parts[:3]) + "/"
                    return
            if call.path.startswith("/testbed/"):
                session.root = "/testbed/"
                return


def rows(path: Path, source: str) -> Iterator[dict[str, Any]]:
    files = (
        sorted(p for p in path.rglob("*") if p.suffix in (".parquet", ".jsonl"))
        if path.is_dir()
        else [path]
    )
    for file in files:
        if source == "pi":
            if file.suffix == ".jsonl":
                yield pi_row(file)
            continue
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
                yield from batch.to_pylist()
        else:
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
