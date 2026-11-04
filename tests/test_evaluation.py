from __future__ import annotations

import copy

import pytest

from arenas import definition, envelopes, spec


def test_profiles_document_matches_the_contract() -> None:
    document = envelopes.document("profiles")
    assert document["version"] == 2
    assert list(document["profiles"]) == ["P1", "P2", "P3", "K", "C"]
    assert [p["arch"] for p in document["profiles"].values()] == [
        "sm89",
        "sm80",
        "sm90",
        "sm75",
        "cpu",
    ]
    assert document["calibration"]["items"] == 1000


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e["profiles"].append("P9"), "unknown profile"),
        (lambda e: e["images"]["engine"].update(image="vllm/vllm-openai:latest"), "latest"),
        (lambda e: e["images"]["engine"].update(digest="sha256:abc"), "64 hex"),
        (lambda e: e["images"].pop("runner"), "runner is required"),
        (lambda e: e["ban"].update(epochs=0), "positive integer"),
        (lambda e: e.update(mode="owner"), "evaluation.mode"),
        (lambda e: e.update(extra=1), "unknown key"),
    ],
)
def test_evaluation_block_is_validated(change, message: str) -> None:
    arena = definition.load("ctx_compress")
    broken = copy.deepcopy(arena)
    change(broken["evaluation"])
    with pytest.raises(ValueError, match=message):
        definition.evaluation(broken)
    assert definition.evaluation(arena) == arena["evaluation"]


def test_ctx_compress_page_states_the_coordinator_rules() -> None:
    page = spec.markdown(spec.build(definition.load("ctx_compress")))
    for line in (
        "reruns 40 SLM calls per miner",
        "all 40 sampled calls run one at a time",
        "10 of them, chosen by a hash of the call, are timed",
        "first token ≤ 4 s up to 32,768 input tokens",
        "first token ≤ 25 s and end to end ≤ 75 s",
        "per token ≤ 20 ms",
        "agree with each other and differ from the miner",
        "apply a 3 epoch hotkey ban",
        "the server never bans directly",
        "within one epoch the miner can appeal",
        "the earliest commit on chain wins",
        "in its own pool and in the full system pool",
        "at most 3 per hotkey per epoch",
    ):
        assert line in page
    for stale in ("Resume", "resume", "confirmations", "validators per miner", "at least 99%"):
        assert stale not in page


def test_miner_mode_renders_section_nine_and_reaches_the_definition() -> None:
    arena = definition.load("ctx_compress")
    page = spec.markdown(spec.build(arena))
    assert "## 9. Evaluation and hardware profiles" in page
    assert "| P1 | Small GPU |" in page
    assert "`vllm/vllm-openai:v0.32.0`" in page
    built = definition.build("ctx_compress", genesis_block=1)
    assert built["definition"]["evaluation"] == arena["evaluation"]
    assert "evaluation" in built["definition"]["spec"]
    with pytest.raises(ValueError, match="settle margin"):
        definition.build("ctx_compress", genesis_block=1, settle_margin=300)
    for arena_id in definition.arena_ids():
        loaded = definition.load(arena_id)
        assert loaded["evaluation"]["mode"] == "miner" and list(loaded["profiles"]) == ["P1"]
    other = {k: v for k, v in arena.items() if k not in ("evaluation", "profiles")}
    assert definition.evaluation(other) is None
    assert "## 9." not in spec.markdown(spec.build(other))
    assert "evaluation" not in spec.build(other)
