from __future__ import annotations

import copy
import math

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
        "a duplicated router in the full system pool",
        "at most 3 per hotkey per epoch",
        "| 60% | 40% |",
        "a validator with a GPU of that profile may rerun the same calls",
        "| No result |",
        "validators keep the last settled weights",
    ):
        assert line in page
    for stale in (
        "Resume",
        "resume",
        "confirmations",
        "validators per miner",
        "at least 99%",
        "20%",
        "S_h =",
        "S_r =",
        "Added wall time",
        "Added latency",
    ):
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


def test_every_arena_pays_two_pools_on_the_checked_profile() -> None:
    for arena_id in definition.arena_ids():
        built = definition.build(arena_id, genesis_block=1, sample_database="0" * 64)
        body = built["definition"]
        assert body["item_counts"] == {"slm": 25, "system": 25}
        assert body["spec"]["scoring"]["shares"] == {"slm": 60, "system": 40}
        assert body["scoring"]["empty_pools"] == "burn"
        assert not set(body["serving"]) & {"cap_schedule", "floor_epochs", "floor_share"}
    arena = definition.load("ctx_compress")
    for counts, message in (
        ({"slm": 25, "system": 25, "harness": 25}, "unknown key 'harness'"),
        ({"slm": 10, "system": 25}, "below scoring.min_items"),
    ):
        with pytest.raises(ValueError, match=message):
            definition.item_counts({**arena, "item_counts": counts})
    broken = copy.deepcopy(arena)
    broken["evaluation"]["profiles"] = ["P1", "P2"]
    with pytest.raises(ValueError, match="no spot check covers"):
        definition.evaluation(broken)


def _tiered() -> dict:
    return {
        "pricing": "tiered",
        "price_schedule": [
            {"input_tokens_below": 50000, "usd_per_request": 0.001},
            {"usd_per_request": 0.004},
        ],
        "revenue_share": 0.8,
    }


def _long_prompt() -> dict:
    return {"above_input_tokens": 50000, "usd_in_per_mtok": 0.3, "usd_out_per_mtok": 1.2}


def test_every_arena_builds_with_an_operator_share_of_eighty_percent() -> None:
    for arena_id in definition.arena_ids():
        arena = definition.load(arena_id)
        built = definition.build(arena_id, genesis_block=1, sample_database="0" * 64)["definition"]
        assert built["serving"] == arena["serving"]
        assert built["serving"]["revenue_share"] == 0.8
        assert built["baseline_models"] == arena["escalation"]["frontier"]


def test_tiered_serving_takes_a_price_schedule() -> None:
    assert definition.checked_serving(_tiered()) == _tiered()
    assert definition.checked_serving(None) == {}
    single = {"pricing": "tiered", "price_schedule": [{"usd_per_request": 0.004}]}
    assert definition.checked_serving(single) == single


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s["price_schedule"][0].update(usd_per_request=0), "positive usd_per_request"),
        (lambda s: s["price_schedule"][1].update(input_tokens_below=200000), "last band without"),
        (lambda s: s["price_schedule"].insert(0, dict(s["price_schedule"][0])), "must rise"),
        (lambda s: s["price_schedule"][0].update(input_tokens_below=2**31), "at most 2147483647"),
        (lambda s: s["price_schedule"][1].update(usd_per_request=math.inf), "finite"),
        (lambda s: s["price_schedule"][1].update(usd_per_request=10**400), "finite"),
        (lambda s: s["price_schedule"][1].update(usd_per_request=10.5), "at most 10"),
        (lambda s: s.update(usd_in_per_mtok=0.01), "no token prices"),
        (lambda s: s.update(pricing="tokens"), "belongs to tiered"),
    ],
)
def test_bad_price_bands_and_mixed_modes_are_refused(change, message: str) -> None:
    serving = _tiered()
    change(serving)
    with pytest.raises(ValueError, match=message):
        definition.checked_serving(serving)


def test_a_long_prompt_tier_validates_and_stays_out_of_the_page_and_the_relay(monkeypatch) -> None:
    arena = definition.load("ctx_compress")
    frontier = copy.deepcopy(arena["escalation"]["frontier"])
    frontier[0]["long_prompt"] = _long_prompt()
    assert definition.checked_frontier(frontier) == frontier
    plain = [{k: v for k, v in model.items() if k != "long_prompt"} for model in frontier]
    with_tier = {**arena, "escalation": {**arena["escalation"], "frontier": frontier}}
    without = {**arena, "escalation": {**arena["escalation"], "frontier": plain}}
    page = spec.markdown(spec.build(with_tier))
    assert page == spec.markdown(spec.build(without))
    assert (
        "anthropic/claude-haiku-5.5 ($0.1 in, $0.5 out per M tokens)<br>"
        "z-ai/glm-5.3-flash ($0.048 in, $0.2 out per M tokens) |"
    ) in page
    monkeypatch.setattr(definition, "load", lambda arena_id: without)
    relay = definition.relay_catalog("ctx_compress")
    monkeypatch.setattr(definition, "load", lambda arena_id: with_tier)
    assert definition.relay_catalog("ctx_compress") == relay
    assert all("long_prompt" not in row for row in relay)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.update(context=200000), "unknown key 'context'"),
        (lambda m: m.update(usd_in_per_mtok="0.1"), "usd_in_per_mtok must be a number"),
        (lambda m: m.update(usd_out_per_mtok=math.inf), "usd_out_per_mtok must be finite"),
        (lambda m: m["long_prompt"].update(usd_in_per_mtok=0, usd_out_per_mtok=0), "above 0"),
        (lambda m: m["long_prompt"].update(usd_per_request=1), "unknown key 'usd_per_request'"),
        (lambda m: m["long_prompt"].update(above_input_tokens=2**31), "at most 2147483647"),
        (lambda m: m["long_prompt"].update(usd_in_per_mtok=1000.5), "at most 1000"),
        (lambda m: m["long_prompt"].update(usd_out_per_mtok=10**400), "must be finite"),
    ],
)
def test_frontier_entries_are_validated(change, message: str) -> None:
    frontier = copy.deepcopy(definition.load("ctx_compress")["escalation"]["frontier"])
    frontier[0]["long_prompt"] = _long_prompt()
    change(frontier[0])
    with pytest.raises(ValueError, match=message):
        definition.checked_frontier(frontier)
