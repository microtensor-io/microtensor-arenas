from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from arenas import envelopes, spec

ROOT = Path(__file__).resolve().parent.parent
ARENA_ID = "sql_gen"
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}\Z")
ARCHIVED = "archived.yaml"
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
OPTIONAL_EVALUATION_KEYS = ("resume",)
MINER_SETTLE_MARGIN = 1800
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
POOLS = ("slm", "system")
SPOT_CHECKED_PROFILES = ("P1",)
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
    "empty_pools": "burn",
    "router": "floors_brier",
    "min_items": 25,
}


def _folders() -> list[Path]:
    return sorted(p.parent for p in ROOT.glob("*/arena.yaml") if ID.match(p.parent.name))


def arena_ids() -> list[str]:
    return [p.name for p in _folders() if not (p / ARCHIVED).is_file()]


def archived_ids() -> list[str]:
    return [p.name for p in _folders() if (p / ARCHIVED).is_file()]


def checked_id(arena_id: Any) -> str:
    if not isinstance(arena_id, str) or not ID.match(arena_id):
        raise ValueError(
            f"arena id {arena_id!r} must be 1 to 24 lowercase letters, digits, underscores or "
            "hyphens, starting with a letter or digit"
        )
    return arena_id


def folder(arena_id: str) -> Path:
    path = ROOT / checked_id(arena_id)
    if not (path / "arena.yaml").is_file():
        raise ValueError(f"no arena folder for {arena_id!r}; known: {', '.join(arena_ids())}")
    return path


def archive(arena_id: str) -> dict[str, Any] | None:
    path = folder(arena_id) / ARCHIVED
    if not path.is_file():
        return None
    held = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), ARCHIVED)
    checked_id(held.get("successor"))
    return held


def raw(arena_id: str) -> dict[str, Any]:
    path = folder(arena_id)
    found = _mapping(yaml.safe_load((path / "arena.yaml").read_text(encoding="utf-8")), "arena")
    if found.get("id") != arena_id:
        raise ValueError(
            f"{arena_id}/arena.yaml names id {found.get('id')!r}; the folder name is the arena id"
        )
    return found


def task(arena_id: str) -> str:
    return str(raw(arena_id)["task"])


def load(arena_id: str) -> dict[str, Any]:
    held = archive(arena_id)
    if held is not None:
        raise ValueError(
            f"{arena_id} is archived and is never redefined or reopened; its successor is "
            f"{held['successor']}"
        )
    path = folder(arena_id)
    arena = raw(arena_id)
    arena["schemas"] = {
        name: json.loads((path / "schema" / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("input", "output")
    }
    arena["envelope"] = envelopes.resolve(arena["envelope"])
    arena["scoring"] = scoring(arena)
    item_counts(arena)
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
    for name in value:
        if name not in SPOT_CHECKED_PROFILES:
            raise ValueError(
                f"evaluation.profiles lists {name}, which no spot check covers yet; list only "
                f"{', '.join(SPOT_CHECKED_PROFILES)}"
            )


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
    for key in block:
        if key not in ("model", "revision", "file", "max_tokens", "max_input_tokens"):
            raise ValueError(f"unknown key {key!r} in {where}")
    for key in ("model", "revision", "file", "max_tokens"):
        if key not in block:
            raise ValueError(f"{where}.{key} is required")
    if not isinstance(block["model"], str) or "/" not in block["model"]:
        raise ValueError(f"{where}.model must be a Hugging Face repository id")
    if not isinstance(block["revision"], str) or not REVISION.match(block["revision"]):
        raise ValueError(f"{where}.revision must be a 40 hex commit sha")
    if not isinstance(block["file"], str) or not block["file"].endswith(".gguf"):
        raise ValueError(f"{where}.file must name a .gguf file")
    _positive(block, "max_tokens", where)
    if "max_input_tokens" in block:
        _positive(block, "max_input_tokens", where)


def _resume(value: Any) -> None:
    where = "evaluation.resume"
    block = _mapping(value, where)
    _exact(block, ("items", "margin", "agent"), where)
    _positive(block, "items", where)
    margin = block["margin"]
    if isinstance(margin, bool) or not isinstance(margin, int | float) or not 0 <= margin < 1:
        raise ValueError(f"{where}.margin must be a fraction")
    agent = block["agent"]
    if agent is None:
        return
    agent = _mapping(agent, f"{where}.agent")
    _exact(agent, ("image", "digest", "model"), f"{where}.agent")
    _image("resume_agent", {"image": agent["image"], "digest": agent["digest"]})
    if not isinstance(agent["model"], str) or not agent["model"]:
        raise ValueError(f"{where}.agent.model names the relay model the agent runs on")


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
        if key not in EVALUATION_KEYS and key not in OPTIONAL_EVALUATION_KEYS:
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
    if "resume" in block:
        _resume(block["resume"])
    if "computed_cost" in block:
        _computed_cost(block["computed_cost"])
    return block


def item_counts(arena: dict[str, Any]) -> dict[str, int]:
    counts = _mapping(arena.get("item_counts"), "item_counts")
    _counts(counts, POOLS, "item_counts")
    floor = arena["scoring"]["min_items"]
    for pool, count in counts.items():
        if count < floor:
            raise ValueError(
                f"item_counts.{pool} is {count}, below scoring.min_items {floor}, so no miner "
                "could be paid"
            )
    return counts


def scoring(arena: dict[str, Any]) -> dict[str, Any]:
    found = dict(SCORING)
    for key, value in (arena.get("scoring") or {}).items():
        if key not in SCORING:
            raise ValueError(f"unknown scoring key {key!r}")
        default = SCORING[key]
        found[key] = {**default, **value} if isinstance(default, dict) and value else value
    return found


SERVING_KEYS = (
    "pricing",
    "usd_per_request",
    "kind",
    "rate",
    "unit",
    "usd_in_per_mtok",
    "usd_out_per_mtok",
    "revenue_share",
    "price_schedule",
)
SERVING_PRICING = ("tokens", "flat", "tiered")
FRONTIER_KEYS = ("model", "provider", "version", "usd_in_per_mtok", "usd_out_per_mtok")
LONG_PROMPT_KEYS = ("above_input_tokens", "usd_in_per_mtok", "usd_out_per_mtok")
MAX_INPUT_TOKENS = 2_147_483_647
MAX_PRICE_PER_MTOK = 1000.0
MAX_PRICE_PER_REQUEST = 10.0
REBATE = 1 / 25


def _finite(value: float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _price_schedule(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("serving.price_schedule is a list of bands")
    previous: int | None = None
    for index, band in enumerate(value):
        band = _mapping(band, "serving.price_schedule band")
        if set(band) - {"input_tokens_below", "usd_per_request"}:
            raise ValueError(
                "a price schedule band carries usd_per_request and optionally input_tokens_below"
            )
        price = band.get("usd_per_request")
        if isinstance(price, bool) or not isinstance(price, int | float) or not price > 0:
            raise ValueError("each price schedule band needs a positive usd_per_request")
        if not _finite(price):
            raise ValueError("each price schedule band needs a finite usd_per_request")
        if price > MAX_PRICE_PER_REQUEST:
            raise ValueError(f"a band's usd_per_request is at most {MAX_PRICE_PER_REQUEST:g}")
        if "input_tokens_below" not in band:
            if index != len(value) - 1:
                raise ValueError("the band without input_tokens_below must be the last one")
            continue
        bound = band["input_tokens_below"]
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise ValueError("input_tokens_below is a positive whole number of tokens")
        if bound > MAX_INPUT_TOKENS:
            raise ValueError(f"input_tokens_below is at most {MAX_INPUT_TOKENS} tokens")
        if previous is not None and bound <= previous:
            raise ValueError("price schedule bands must rise in input_tokens_below")
        previous = bound
    if "input_tokens_below" in value[-1]:
        raise ValueError("the price schedule needs a last band without input_tokens_below")


def checked_serving(serving: Any) -> dict[str, Any]:
    serving = dict(_mapping(serving or {}, "serving"))
    unknown = set(serving) - set(SERVING_KEYS)
    if unknown:
        raise ValueError(f"serving carries unknown keys: {', '.join(sorted(unknown))}")
    pricing = serving.get("pricing", "tokens")
    if pricing not in SERVING_PRICING:
        raise ValueError("serving.pricing is tokens, flat or tiered")
    if pricing == "flat":
        price = serving.get("usd_per_request")
        if isinstance(price, bool) or not isinstance(price, int | float) or price <= 0:
            raise ValueError("flat serving needs a positive usd_per_request")
        if not _finite(price):
            raise ValueError("flat serving needs a finite usd_per_request")
        if price > MAX_PRICE_PER_REQUEST:
            raise ValueError(f"flat usd_per_request is at most {MAX_PRICE_PER_REQUEST:g}")
        if "usd_in_per_mtok" in serving or "usd_out_per_mtok" in serving:
            raise ValueError("flat serving carries no token prices")
    elif "usd_per_request" in serving:
        raise ValueError("usd_per_request belongs to flat serving")
    if pricing == "tiered":
        if "usd_in_per_mtok" in serving or "usd_out_per_mtok" in serving:
            raise ValueError("tiered serving carries no token prices")
        if "price_schedule" not in serving:
            raise ValueError("tiered serving needs a price_schedule")
        _price_schedule(serving["price_schedule"])
    elif "price_schedule" in serving:
        raise ValueError("price_schedule belongs to tiered serving")
    if "revenue_share" in serving and (
        not isinstance(serving["revenue_share"], int | float)
        or isinstance(serving["revenue_share"], bool)
        or not 0 <= serving["revenue_share"] < 1 - REBATE
    ):
        raise ValueError(
            "serving.revenue_share is the operators' fraction of billing and must leave "
            f"Microtensor more than the {REBATE:g} rebate"
        )
    return serving


def _rate(block: dict[str, Any], key: str, where: str) -> None:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        raise ValueError(f"{where}.{key} must be a number of at least 0")
    if not _finite(value):
        raise ValueError(f"{where}.{key} must be finite")


def _long_prompt(value: Any, where: str) -> dict[str, Any]:
    block = dict(_mapping(value, where))
    _exact(block, LONG_PROMPT_KEYS, where)
    _positive(block, "above_input_tokens", where)
    if block["above_input_tokens"] > MAX_INPUT_TOKENS:
        raise ValueError(f"{where}.above_input_tokens must be at most {MAX_INPUT_TOKENS}")
    for key in ("usd_in_per_mtok", "usd_out_per_mtok"):
        _rate(block, key, where)
        if block[key] > MAX_PRICE_PER_MTOK:
            raise ValueError(f"{where}.{key} must be at most {MAX_PRICE_PER_MTOK:g}")
    if not block["usd_in_per_mtok"] and not block["usd_out_per_mtok"]:
        raise ValueError(f"{where} needs a price above 0 on at least one side")
    return block


def _listed(value: Any, where: str) -> list[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{where} must be a list")


def checked_frontier(entries: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for index, entry in enumerate(_listed(entries, "escalation.frontier")):
        where = f"escalation.frontier[{index}]"
        block = dict(_mapping(entry, where))
        for key in block:
            if key not in FRONTIER_KEYS and key != "long_prompt":
                raise ValueError(f"unknown key {key!r} in {where}")
        for key in ("model", "provider", "version"):
            if not isinstance(block.get(key), str) or not block[key].strip():
                raise ValueError(f"{where}.{key} must be a non empty string")
        for key in ("usd_in_per_mtok", "usd_out_per_mtok"):
            _rate(block, key, where)
        if "long_prompt" in block:
            block["long_prompt"] = _long_prompt(block["long_prompt"], f"{where}.long_prompt")
        found.append(block)
    return found


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
        "baseline_models": checked_frontier(arena["escalation"]["frontier"]),
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
            },
            "harness": {
                key: harness[key]
                for key in (
                    "request_timeout_s",
                    "max_steps",
                    "max_total_tokens",
                    "max_model_calls",
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
        "serving": checked_serving(arena.get("serving")),
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
            "max_tokens": 3072,
        }
        for b in load(arena_id)["escalation"]["frontier"]
    ]
