import json
from pathlib import Path

from arenas.compress import build, canon, hidden, sources, synth


def count(text: str) -> int:
    return max(1, len(text) // 4)


def test_sessions_follow_their_seed():
    first = [row["session_id"] for row in synth.rows("alpha", 2, (20_000, 24_000), workers=1)]
    again = [row["session_id"] for row in synth.rows("alpha", 2, (20_000, 24_000), workers=1)]
    other = [row["session_id"] for row in synth.rows("beta", 2, (20_000, 24_000), workers=1)]
    assert first == again
    assert not set(first) & set(other)


def test_dev_items_pass_the_check():
    pool, report = build.generate(
        "ctx_compress", 3, 2, [("synth", Path("4@20000-26000"))], min_tokens=2_000, count=count
    )
    assert report["sessions"] == 2
    assert any(item["twin_of"] for item in pool.items)
    assert all(item["label"]["source"] == "synth" for item in pool.items)
    assert all(item["label"]["image"].startswith("synth:") for item in pool.items)
    assert build.check(pool) == []


def test_hidden_items_keep_the_seed_secret(tmp_path):
    seed = hidden.new_seed()
    report = hidden.build_hidden(
        "ctx_compress",
        seed,
        [("synth", Path("6@20000-26000"))],
        tmp_path,
        counts={"slm": 1, "system": 1},
        min_tokens=2_000,
        count=count,
    )
    assert report["per_test"] == {"slm": 1, "system": 1}
    assert seed not in (tmp_path / "items.jsonl").read_text(encoding="utf-8")
    assert seed not in (tmp_path / "report.json").read_text(encoding="utf-8")


def _pi_file(tmp_path):
    def entry(uid, parent, message):
        return {"type": "message", "id": uid, "parentId": parent, "message": message}

    call = {
        "type": "toolCall",
        "id": "c1",
        "name": "read",
        "arguments": {"path": "/repo/app/main.py"},
    }
    rows = [
        {"type": "session", "version": 3, "id": "pi-1", "cwd": "/repo", "timestamp": "2026-01-05"},
        entry("a", None, {"role": "user", "content": [{"type": "text", "text": "Fix the export"}]}),
        entry("x", "a", {"role": "assistant", "content": [{"type": "text", "text": "dropped"}]}),
        entry(
            "b",
            "a",
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hidden"}, call],
            },
        ),
        entry(
            "c",
            "b",
            {
                "role": "toolResult",
                "toolCallId": "c1",
                "content": [{"type": "text", "text": "def main():\n    pass"}],
                "isError": False,
            },
        ),
        entry("d", "c", {"role": "assistant", "content": [{"type": "text", "text": "Read it."}]}),
    ]
    path = tmp_path / "pi-1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pi_sessions_follow_the_active_branch(tmp_path):
    session = sources.parse(sources.pi_row(_pi_file(tmp_path)), "pi")
    assert [step.role for step in session.steps] == ["user", "agent", "tool", "agent"]
    assert session.steps[1].calls[0].kind == "read"
    assert session.relative(session.steps[1].calls[0].path) == "app/main.py"
    rendered = canon.render(session, len(session.steps))
    assert "hidden" not in rendered and "dropped" not in rendered
    assert "[tool_use kind=read name=read path=app/main.py]" in rendered


def test_attached_probes_must_be_in_the_session():
    pool, _ = build.generate(
        "ctx_compress", 3, 1, [("synth", Path("4@20000-26000"))], min_tokens=2_000, count=count
    )
    item = pool.items[0]
    item["label"]["attached"] = [
        {"id": "kept", "keys": [item["task"]["user"][20:60]], "in_session": True},
        {"id": "lost", "keys": ["never in any session 9f2c"], "in_session": True},
    ]
    assert build.check(pool) == [f"{item['item_id']}: attached probe lost is not in the session"]
