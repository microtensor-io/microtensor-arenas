from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

SHARED = Path(__file__).resolve().parent.parent / "shared"
MAXIMA = ("memory_gb", "max_input_tokens", "max_output_tokens", "ttft_p95_ms", "tpot_p95_ms")
MINIMA = ("min_tokens_per_s",)
SUBSETS = ("precision", "formats")
LIMITS = (*MAXIMA, *MINIMA, *SUBSETS, "parameters")


@cache
def document(kind: str) -> dict[str, Any]:
    found: dict[str, Any] = yaml.safe_load((SHARED / f"{kind}.yaml").read_text(encoding="utf-8"))
    return found


def group(name: str) -> dict[str, Any]:
    groups = document("envelopes")["groups"]
    if name not in groups:
        raise ValueError(f"unknown envelope group {name!r}; known: {', '.join(groups)}")
    found: dict[str, Any] = groups[name]
    if found.get("reserved"):
        raise ValueError(f"envelope group {name} is reserved")
    if not found.get("open"):
        raise ValueError(f"envelope group {name} is not open; it opens in {found.get('opens')}")
    return found


def _tighter(name: str, key: str, value: Any, base: Any) -> None:
    if base is None:
        raise ValueError(f"{name} has no {key} to override")
    if key in MAXIMA and value > base:
        raise ValueError(f"{key} {value} is looser than {name}'s {base}")
    if key in MINIMA and value < base:
        raise ValueError(f"{key} {value} is looser than {name}'s {base}")
    if key in SUBSETS and not set(value) <= set(base):
        raise ValueError(f"{key} {value} is not within {name}'s {base}")
    if key == "parameters":
        if value.get("max", 0) > base.get("max", 0):
            raise ValueError(f"parameters max {value['max']} is looser than {name}'s {base['max']}")
        if value.get("min", 0) < base.get("min", 0):
            raise ValueError(f"parameters min {value['min']} is looser than {name}'s {base['min']}")


def resolve(envelope: dict[str, Any]) -> dict[str, Any]:
    name = envelope["group"]
    base = group(name)
    overrides = dict(envelope.get("overrides") or {})
    limits = {key: base[key] for key in LIMITS if key in base}
    for key, value in overrides.items():
        if key not in LIMITS:
            raise ValueError(f"{key} is not an envelope limit")
        _tighter(name, key, value, base.get(key))
        limits[key] = {**base[key], **value} if key == "parameters" else value
    shared = document("envelopes")
    return {
        "group": name,
        "name": base["name"],
        "version": shared["version"],
        "docs": shared["docs"],
        "runtime": base["runtime"],
        "device": base["device"],
        "reference": base["reference"],
        "violation_share": shared["violation_share"],
        "limits": limits,
        "overrides": overrides,
    }


def budgets(resolved: dict[str, Any]) -> dict[str, Any]:
    limits = resolved["limits"]
    memory = limits["memory_gb"] * 1024**3
    found: dict[str, Any] = {
        "ttft_ms": limits["ttft_p95_ms"],
        "memory_gb": limits["memory_gb"],
        "memory_bytes": memory,
        "max_input_tokens": limits["max_input_tokens"],
        "max_output_tokens": limits["max_output_tokens"],
        "min_parameters": limits["parameters"].get("min", 0),
        "max_parameters": limits["parameters"]["max"],
    }
    if "tpot_p95_ms" in limits:
        found["tpot_ms"] = limits["tpot_p95_ms"]
    if "min_tokens_per_s" in limits:
        found["min_tokens_per_s"] = limits["min_tokens_per_s"]
    if resolved["device"] == "gpu":
        found["vram_gb"] = limits["memory_gb"]
        found["vram_bytes"] = memory
    return found


def serving_class(resolved: dict[str, Any]) -> str:
    return str(group(resolved["group"]).get("serving_class") or resolved["group"])


def lower(name: str) -> list[str]:
    groups = document("envelopes")["groups"]
    names = list(groups)
    return [n for n in names[: names.index(name)] if groups[n].get("open")]


def base(model: str) -> dict[str, Any]:
    for entry in document("bases")["bases"]:
        if entry["model"] == model:
            found: dict[str, Any] = entry
            return found
    raise ValueError(f"{model} is not on the shared base list")
