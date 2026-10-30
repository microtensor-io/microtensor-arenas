from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from arenas import envelopes, spec

ROOT = Path(__file__).resolve().parent.parent
ARENA_ID = "sql_gen"
NETWORKS = ("testnet", "mainnet")
EVALUATION_MODES = ("miner",)
GRADER_TYPES = ("pure_cpu", "model_assisted")
IMAGES = ("engine", "runner", "sandbox", "replayer", "llama")
RELEASE = {"inputs": "window_open", "labels": "after_freeze_2"}
EVALUATION_KEYS = (
    "mode",
    "profiles",
    "grader_type",
    "release",
    "images",
    "probe",
    "spot_check",
    "ban",
    "canaries",
    "relay_allowance_micros",
    "computed_cost",
)
MINER_SETTLE_MARGIN = 1800
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
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
    "min_items": 400,
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
    block = evaluation(arena)
    if block is not None:
        arena["profiles"] = {name: profiles()[name] for name in block.get("profiles") or []}
    return arena


def profiles() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = envelopes.document("profiles")["profiles"]
    return found


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{where} must be a mapping")


def _exact(block: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    for key in block:
        if key not in keys:
            raise ValueError(f"unknown key {key!r} in {where}")
    for key in keys:
        if key not in block:
            raise ValueError(f"{where}.{key} is required")


def _positive(block: dict[str, Any], key: str, where: str, *, zero: bool = False) -> None:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        kind = "a non negative" if zero else "a positive"
        raise ValueError(f"{where}.{key} must be {kind} integer")


def _profiles(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation.profiles must list at least one profile")
    known = profiles()
    for name in value:
        if name not in known:
            raise ValueError(
                f"unknown profile {name!r} in evaluation.profiles; known: {', '.join(known)}"
            )
    if len(set(value)) != len(value):
        raise ValueError("evaluation.profiles repeats a profile")


def _image(name: str, entry: Any) -> None:
    where = f"evaluation.images.{name}"
    block = _mapping(entry, where)
    _exact(block, ("image", "digest"), where)
    image = block["image"]
    if not isinstance(image, str) or "@" in image:
        raise ValueError(f"{where}.image must be name:tag; the digest goes in {where}.digest")
    repository, _, tag = image.rpartition(":")
    if not repository or not tag or "/" in tag:
        raise ValueError(f"{where}.image {image!r} needs a tag")
    if tag == "latest":
        raise ValueError(f"{where}.image must never use the latest tag")
    digest = block["digest"]
    if not isinstance(digest, str) or not DIGEST.match(digest):
        raise ValueError(f"{where}.digest must be sha256: followed by 64 hex characters")


def _images(value: Any, *, complete: bool) -> None:
    block = _mapping(value, "evaluation.images")
    for name in block:
        if name not in IMAGES:
            raise ValueError(
                f"unknown image {name!r} in evaluation.images; one of {', '.join(IMAGES)}"
            )
    if complete:
        for name in IMAGES:
            if name not in block:
                raise ValueError(f"evaluation.images.{name} is required")
    for name, entry in block.items():
        _image(name, entry)


def _probe(value: Any) -> None:
    where = "evaluation.probe"
    block = _mapping(value, where)
    _exact(block, ("model", "revision", "file", "max_tokens"), where)
    if not isinstance(block["model"], str) or "/" not in block["model"]:
        raise ValueError(f"{where}.model must be a Hugging Face repository id")
    if not isinstance(block["revision"], str) or not REVISION.match(block["revision"]):
        raise ValueError(f"{where}.revision must be a 40 hex commit sha")
    if not isinstance(block["file"], str) or not block["file"].endswith(".gguf"):
        raise ValueError(f"{where}.file must name a .gguf file")
    _positive(block, "max_tokens", where)


def _counts(value: Any, keys: tuple[str, ...], where: str) -> None:
    block = _mapping(value, where)
    _exact(block, keys, where)
    for key in keys:
        _positive(block, key, where)


def _computed_cost(value: Any) -> None:
    where = "evaluation.computed_cost"
    block = _mapping(value, where)
    _exact(block, ("bytes_per_weight", "cap_units_per_item"), where)
    weights = _mapping(block["bytes_per_weight"], f"{where}.bytes_per_weight")
    if not weights:
        raise ValueError(f"{where}.bytes_per_weight must list at least one precision")
    for precision, amount in weights.items():
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError(f"{where}.bytes_per_weight.{precision} must be a positive number")
    _positive(block, "cap_units_per_item", where, zero=True)


def evaluation(arena: dict[str, Any]) -> dict[str, Any] | None:
    if "evaluation" not in arena:
        return None
    block = _mapping(arena["evaluation"], "evaluation")
    for key in block:
        if key not in EVALUATION_KEYS:
            raise ValueError(f"unknown key {key!r} in evaluation")
    mode = block.get("mode")
    if mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation.mode must be one of {', '.join(EVALUATION_MODES)}")
    miner = mode == "miner"
    if miner:
        for key in EVALUATION_KEYS:
            if key not in block:
                raise ValueError(f"evaluation.{key} is required")
    if "profiles" in block:
        _profiles(block["profiles"])
    if "grader_type" in block and block["grader_type"] not in GRADER_TYPES:
        raise ValueError(f"evaluation.grader_type must be one of {', '.join(GRADER_TYPES)}")
    if "release" in block and _mapping(block["release"], "evaluation.release") != RELEASE:
        raise ValueError(
            "evaluation.release must be inputs: window_open and labels: after_freeze_2"
        )
    if "images" in block:
        _images(block["images"], complete=miner)
    if "probe" in block:
        _probe(block["probe"])
    if "spot_check" in block:
        _counts(
            block["spot_check"],
            ("min_calls", "validators_per_miner", "reference_calls"),
            "evaluation.spot_check",
        )
    if "ban" in block:
        _counts(block["ban"], ("epochs", "confirmations"), "evaluation.ban")
    if "canaries" in block:
        _counts(block["canaries"], ("count",), "evaluation.canaries")
    if "relay_allowance_micros" in block:
        _positive(block, "relay_allowance_micros", "evaluation")
    if "computed_cost" in block:
        _computed_cost(block["computed_cost"])
    return block


def scoring(arena: dict[str, Any]) -> dict[str, Any]:
    found = dict(SCORING)
    for key, value in (arena.get("scoring") or {}).items():
        if key not in SCORING:
            raise ValueError(f"unknown scoring key {key!r}")
        default = SCORING[key]
        found[key] = {**default, **value} if isinstance(default, dict) and value else value
    return found


SERVING_KEYS = (
    "rate",
    "unit",
    "usd_in_per_mtok",
    "usd_out_per_mtok",
    "cap_schedule",
    "floor_epochs",
    "floor_share",
)
MAX_FLOOR_SHARE = 0.5


def _schedule(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("serving.cap_schedule is a list of bands")
    previous: float | None = None
    for index, band in enumerate(value):
        band = _mapping(band, "serving.cap_schedule band")
        if set(band) - {"billing_usd_below", "cap"}:
            raise ValueError("a cap schedule band carries cap and optionally billing_usd_below")
        if not isinstance(band.get("cap"), int | float) or not 0 <= band["cap"] <= 1:
            raise ValueError("each cap schedule band needs a cap between 0 and 1")
        bound = band.get("billing_usd_below")
        if bound is None:
            if index != len(value) - 1:
                raise ValueError("the band without billing_usd_below must be the last one")
            continue
        if not isinstance(bound, int | float) or bound < 0:
            raise ValueError("billing_usd_below is an amount in USD")
        if previous is not None and bound <= previous:
            raise ValueError("cap schedule bands must rise in billing_usd_below")
        previous = float(bound)
    if "billing_usd_below" in value[-1]:
        raise ValueError("the cap schedule needs a last band without billing_usd_below")


def _serving(arena: dict[str, Any]) -> dict[str, Any]:
    serving = dict(arena.get("serving") or {})
    unknown = set(serving) - set(SERVING_KEYS)
    if unknown:
        raise ValueError(f"serving carries unknown keys: {', '.join(sorted(unknown))}")
    if "cap_schedule" in serving:
        _schedule(serving["cap_schedule"])
    if "floor_epochs" in serving and (
        not isinstance(serving["floor_epochs"], int) or serving["floor_epochs"] < 0
    ):
        raise ValueError("serving.floor_epochs is a whole number of epochs")
    if "floor_share" in serving and (
        not isinstance(serving["floor_share"], int | float)
        or not 0 <= serving["floor_share"] <= MAX_FLOOR_SHARE
    ):
        raise ValueError(f"serving.floor_share is between 0 and {MAX_FLOOR_SHARE}")
    return serving


def _bases(arena: dict[str, Any]) -> list[dict[str, Any]]:
    group = arena["envelope"]["group"]
    reference = arena["slm"].get("reference")
    if not isinstance(reference, dict) or set(reference) != {"model", "revision"}:
        raise ValueError("slm.reference names exactly one model with its revision")
    names = list(arena["slm"]["bases"])
    if len(names) != 1:
        raise ValueError(f"an arena names exactly one base model, not {len(names)}")
    if reference["model"] != names[0]:
        raise ValueError("the reference model is the base model")
    found = []
    for model in names:
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
        "slm_model_class": slm.get("model_class", "AutoModelForCausalLM"),
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
        "serving": _serving(arena),
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
            raise ValueError("sql_gen needs the sha256 of its sample database")
        body["sample_database"] = sample_database
    timing = arena["timing"]
    if arena.get("evaluation"):
        body["evaluation"] = arena["evaluation"]
        if arena["evaluation"]["mode"] == "miner" and timing["settle_margin"] < MINER_SETTLE_MARGIN:
            raise ValueError(
                f"the arena needs a settle margin of at least {MINER_SETTLE_MARGIN} blocks "
                f"(6 h) so grading fits; {arena['id']} has {timing['settle_margin']}"
            )
    return {
        "id": arena["id"],
        "title": arena["title"],
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
