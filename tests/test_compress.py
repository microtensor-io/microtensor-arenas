from pathlib import Path

from arenas.compress import build, hidden, synth


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
