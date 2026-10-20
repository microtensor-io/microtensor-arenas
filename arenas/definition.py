from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from arenas import envelopes, spec

ROOT = Path(__file__).resolve().parent.parent
ARENA_ID = "sql_gen_s1"
NETWORKS = ("testnet", "mainnet")
ITEM_COUNTS = {"slm": 400, "harness": 400, "router": 250, "escalation": 120, "system": 400}
SCORING: dict[str, Any] = {
    "wilson_z": 1.645,
    "twin": {"tolerance": 0.03, "slope": None},
    "efficiency": {
        "form": "log",
        "range": 0.25,
        "weights": {"ttft": 0.4, "memory": 0.3, "parameters": 0.2, "tpot": 0.1},
    },
    "edge_bonus": None,
    "envelope_rule": "p95",
    "cap": 1.0,
    "empty_pools": "redistribute",
    "router": "floors_brier",
}


def arena_ids() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/arena.yaml"))


def folder(arena_id: str) -> Path:
    path = ROOT / arena_id
    if not (path / "arena.yaml").is_file():
        raise ValueError(f"no arena folder for {arena_id!r}; known: {', '.join(arena_ids())}")
    return path


def load(arena_id: str) -> dict[str, Any]:
    path = folder(arena_id)
    arena = yaml.safe_load((path / "arena.yaml").read_text(encoding="utf-8"))
    arena["schemas"] = {
        name: json.loads((path / "schema" / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("input", "output")
    }
    arena["envelope"] = envelopes.resolve(arena["envelope"])
    arena["scoring"] = scoring(arena)
    arena["slm"] = {**arena["slm"], "bases": _bases(arena)}
    return arena


def scoring(arena: dict[str, Any]) -> dict[str, Any]:
    found = dict(SCORING)
    for key, value in (arena.get("scoring") or {}).items():
        if key not in SCORING:
            raise ValueError(f"unknown scoring key {key!r}")
        default = SCORING[key]
        found[key] = {**default, **value} if isinstance(default, dict) and value else value
    return found


def _bases(arena: dict[str, Any]) -> list[dict[str, Any]]:
    group = arena["envelope"]["group"]
    found = []
    for model in arena["slm"]["bases"]:
        entry = envelopes.base(model)
        if group not in entry["groups"]:
            raise ValueError(f"{model} does not fit envelope group {group}")
        found.append(entry)
    return found


def _options(arena: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(arena)
    for key, value in overrides.items():
        if value in (None, ""):
            continue
        if key == "network":
            merged["network"] = value
        elif key == "dataset":
            merged["slm"] = {**merged["slm"], "dataset": value}
        elif key == "reference_slm_revision":
            merged["slm"] = {
                **merged["slm"],
                "reference": {**merged["slm"]["reference"], "revision": value},
            }
        elif key in ("fee_tao", "fee_coldkey"):
            merged["fee"] = {**merged["fee"], key.removeprefix("fee_"): value}
        elif key in ("epoch_blocks", "commit_blocks", "settle_margin"):
            merged["timing"] = {**merged["timing"], key: value}
    if merged["network"] not in NETWORKS:
        raise ValueError(f"network must be one of {', '.join(NETWORKS)}")
    return merged


def build(
    arena_id: str, *, genesis_block: int, sample_database: str = "", **overrides: Any
) -> dict[str, Any]:
    arena = _options(load(arena_id), overrides)
    slm = arena["slm"]
    envelope = arena["envelope"]
    harness = arena["harness"]
    router = arena["router"]
    references: dict[str, Any] = {
        "slm": slm["reference"]["model"],
        "slm_revision": slm["reference"]["revision"],
    }
    if router.get("reference"):
        references["router"] = router["reference"]
    body: dict[str, Any] = {
        "modality": arena["category"].lower(),
        "task": arena["task"],
        "aliases": arena.get("aliases", []),
        "slm_model_class": "AutoModelForCausalLM",
        "allowed_licenses": sorted({b["licence"] for b in slm["bases"]}),
        "allowed_bases": [b["model"] for b in slm["bases"]],
        "allowed_formats": envelope["limits"]["precision"],
        "schemas": arena["schemas"],
        "primary_metric": slm["primary_metric"],
        "success_rule": slm["success_rule"],
        "grader": slm["grader"],
        "twin_rule": slm["twin_rule"],
        "baseline_models": arena["escalation"]["frontier"],
        "references": references,
        "tools": harness["tools"],
        "floors": arena["floors"],
        "envelope": envelope,
        "budgets": envelopes.budgets(envelope),
        "scoring": arena["scoring"],
        "limits": {
            "router": {
                "max_parameters": router["max_parameters"],
                "max_input_tokens": router["max_input_tokens"],
                "latency_gate_ms": router["latency_gate_ms"],
            },
            "harness": {
                key: harness[key]
                for key in (
                    "request_timeout_s",
                    "max_steps",
                    "max_total_tokens",
                    "max_model_calls",
                    "added_wall_p95_ms",
                    "faults",
                )
                if key in harness
            },
        },
        "r_class": arena["r_class"],
        "item_counts": arena["item_counts"],
        "slm_prompt": slm["prompt"],
        "generation": {
            "stop": slm["stop"],
            "max_tokens": envelope["limits"]["max_output_tokens"],
        },
        "fee_tao": arena["fee"]["tao"],
        "fee_coldkey": arena["fee"]["coldkey"],
        "hardware_classes": [envelopes.serving_class(envelope)],
        "chat": arena["chat"],
        "serving": arena.get("serving", {}),
        "spec": spec.build(arena),
        "guide": None,
    }
    if slm.get("dataset"):
        body["dataset"] = slm["dataset"]
    if arena.get("grading"):
        body["grading"] = arena["grading"]
    if arena.get("source"):
        body["source"] = arena["source"]
    if arena["task"] == "sql_generation":
        if not sample_database:
            raise ValueError("sql_gen_s1 needs the sha256 of its sample database")
        body["sample_database"] = sample_database
    timing = arena["timing"]
    return {
        "id": arena["id"],
        "title": arena["title"],
        "season": arena["season"],
        "network": arena["network"],
        "genesis_block": genesis_block,
        "epoch_blocks": timing["epoch_blocks"],
        "commit_blocks": timing["commit_blocks"],
        "settle_margin": timing["settle_margin"],
        "definition": body,
    }


def relay_catalog(arena_id: str = ARENA_ID) -> list[dict[str, Any]]:
    return [
        {
            "model_id": b["model"],
            "provider": b["provider"],
            "provider_model": b["version"],
            "usd_in_per_mtok": b["usd_in_per_mtok"],
            "usd_out_per_mtok": b["usd_out_per_mtok"],
            "max_tokens": 1024,
        }
        for b in load(arena_id)["escalation"]["frontier"]
    ]
