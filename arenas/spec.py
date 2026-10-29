from __future__ import annotations

from typing import Any

TOKEN_ALLOWANCE = 1.5
BRIER_WEIGHT = 0.2
ROUTER_MEAN_WEIGHT = 0.7
ROUTER_MIN_WEIGHT = 0.3
ESCALATION_MARGIN = 0.02
COST_WEIGHT = 0.7
LATENCY_WEIGHT = 0.3
MIN_ITEMS = 400
RESAMPLES = 200
POOL_SHARE = 0.2
RANK_PAYOUT = (40, 25, 15, 12, 8)
LADDER_STEP = 0.01
COMPONENTS = ("Specialised SLM", "Harness", "Router", "Escalation repair", "Full system")
WEIGHT_NAMES = {"ttft": "TTFT", "memory": "mem", "parameters": "N", "tpot": "TPOT"}
BLOCKS_PER_HOUR = 300
MODES = {
    "validator": "validator: every validator runs every miner's model itself",
    "miner": (
        "miner: each miner runs its own evaluation on its own GPU and uploads a transcript; "
        "validators replay it on CPU and spot check sampled calls on the same hardware profile"
    ),
}
GRADERS = {"pure_cpu": "pure CPU", "model_assisted": "model assisted"}
LIMIT_ROWS = {
    "max_input_tokens": ("Max input", "{:,} tokens"),
    "max_output_tokens": ("Max output", "{:,} tokens"),
    "ttft_p95_ms": ("First token (p95)", "≤ {:,} ms"),
    "tpot_p95_ms": ("Per token (p95)", "≤ {:,} ms"),
    "min_tokens_per_s": ("Generation", "≥ {:g} tokens/s"),
}


def _int(value: int) -> str:
    return f"{value:,}"


def _floors(floors: list[float]) -> str:
    return " · ".join(f"{f:g}" for f in floors)


def _price(model: dict[str, Any]) -> str:
    return f"${model['usd_in_per_mtok']:g} in, ${model['usd_out_per_mtok']:g} out per M tokens"


def _billions(value: int) -> str:
    return f"{value / 1e9:g}B"


def limit_row(key: str, value: Any, device: str) -> list[str]:
    if key == "parameters":
        low = value.get("min")
        span = f"{_billions(low)} to " if low else "up to "
        return ["Parameters", span + _billions(value["max"])]
    if key in ("precision", "formats"):
        return [key.capitalize(), ", ".join(map(str, value))]
    if key == "image":
        parts = []
        if "max_long_side_px" in value:
            parts.append(f"≤ {_int(value['max_long_side_px'])} px long side")
        if "max_pixels" in value:
            parts.append(f"≤ {_int(value['max_pixels'])} pixels")
        if "pages_per_call" in value:
            parts.append(f"{value['pages_per_call']} page per call")
        if "page_time_p95_ms" in value:
            parts.append(f"≤ {value['page_time_p95_ms'] / 1000:g} s per page (p95)")
        return ["Page image", ", ".join(parts)]
    if key == "memory_gb":
        return ["GPU memory" if device == "gpu" else "Peak memory", f"≤ {value:g} GB"]
    name, template = LIMIT_ROWS[key]
    return [name, template.format(value)]


def _gate(envelope: dict[str, Any]) -> str:
    limits = envelope["limits"]
    speed = (
        f"TPOT p95 ≤ {_int(limits['tpot_p95_ms'])} ms"
        if "tpot_p95_ms" in limits
        else f"generation ≥ {limits['min_tokens_per_s']} tokens/s"
    )
    return (
        f"G = 1 if TTFT p95 ≤ {_int(limits['ttft_p95_ms'])} ms, {speed}, peak memory ≤ "
        f"{limits['memory_gb']} GB and the weights pass provenance against the declared base, "
        "otherwise 0"
    )


def mechanism(scoring: dict[str, Any], envelope: dict[str, Any]) -> list[dict[str, str]]:
    z = scoring["wilson_z"]
    twin = scoring["twin"]
    efficiency = scoring["efficiency"]
    weights = efficiency["weights"]
    edge = scoring.get("edge_bonus")
    per_item = scoring["envelope_rule"] == "per_item"
    items: list[dict[str, str]] = [
        {
            "name": f"Wilson lower bound on the pass rate (n items, pass rate p, z = {z:g})",
            "formula": "LB = (p + z²/2n − z·√(p(1 − p)/n + z²/4n²)) / (1 + z²/n)",
        },
        {
            "name": "Twin factor (gap is the pass rate difference between original items and their twins)",
            "formula": (
                f"twin = 1 − min(1, max(0, gap − {twin['tolerance']:g}) / {twin['slope']:g})"
                if twin.get("slope")
                else f"twin = 1 − max(0, gap − {twin['tolerance']:g})"
            ),
        },
    ]
    if per_item:
        share = 100 - envelope["violation_share"] * 100
        items += [
            {
                "name": "Provenance gate",
                "formula": (
                    "G = 1 if the fingerprint matches the declared base and the system beats that "
                    "base, otherwise 0"
                ),
            },
            {
                "name": "Envelope pass",
                "formula": f"P = 1 if within the envelope on at least {share:g}% of items, otherwise 0",
            },
        ]
    else:
        items.append({"name": "Gate", "formula": _gate(envelope)})
    terms = [(weights[key], WEIGHT_NAMES[key]) for key in ("ttft", "memory", "parameters", "tpot")]
    if efficiency["form"] == "linear":
        items.append(
            {
                "name": "Efficiency (ref is the reference baseline, x is TTFT, memory, parameters or TPOT)",
                "formula": "e_x = min(1, max(0, (ref − miner) / ref))\nE = "
                + " + ".join(f"{w:g}·e_{n}" for w, n in terms),
            }
        )
    else:
        items.append(
            {
                "name": "Efficiency against the reference model",
                "formula": "E = clamp(−1, 1, "
                + " + ".join(f"{w:g}·ln({n}_ref/{n})" for w, n in terms)
                + ")",
            }
        )
    bonus = ""
    if edge:
        entry = {
            "name": "Edge bonus",
            "formula": (
                f"B = {edge['per_group']:g} for each lower envelope group passed, "
                f"max {edge['max']:g}"
            ),
        }
        if edge.get("applies") is False:
            entry["note"] = "0 in this arena: its inputs are longer than every lower group allows."
        items.append(entry)
        bonus = " + B"
    inner = f"G · {'P · ' if per_item else ''}LB · twin · (1 + {efficiency['range']:g}·E{bonus})"
    items += [
        {
            "name": "SLM score",
            "formula": f"S_slm = clamp(0, 1, {inner})"
            if scoring.get("cap")
            else f"S_slm = {inner}",
        },
        {
            "name": "Paid when",
            "formula": (
                f"n ≥ {scoring['min_items']} and the paired lower bound of (miner − reference model) per item, "
                "clustered by family, is above 0"
            ),
        },
    ]
    return items


def _harness_rows(harness: dict[str, Any]) -> list[list[str]]:
    if harness.get("rows"):
        return [list(map(str, row)) for row in harness["rows"]]
    return [
        ["Endpoint", "POST /harness/run"],
        ["Input", '{"item_id", "task"} with the task fields from the input schema'],
        ["Output", '{"answer": ...} in the task output format'],
        ["Tools", ", ".join(harness["tools"]) or "none"],
        ["Sandbox", harness["sandbox"]],
        ["Request timeout", f"{harness['request_timeout_s']} s"],
        ["Code size", f"≤ {harness['code_limit_mb']} MB with escalation"],
        ["Limits", f"{harness['max_steps']} steps, {_int(harness['max_total_tokens'])} tokens"],
        ["Baselines", "bare model under the arena prompt, reference harness and its variants"],
    ]


def _router_rows(router: dict[str, Any]) -> list[list[str]]:
    if router.get("rows"):
        return [list(map(str, row)) for row in router["rows"]]
    return [
        ["Model", "sequence classifier with one output"],
        ["Max size", f"≤ {_int(router['max_parameters'])} parameters"],
        ["Max input", f"{_int(router['max_input_tokens'])} tokens"],
        ["Request", 'POST /decide {"input": {"query", "draft"}}'],
        ["Response", '{"answers": [{"label": "draft_correct", "probability": p}]}'],
        ["Latency (p95)", f"< {_int(router['latency_gate_ms'])} ms"],
        ["Drafts", "your model, the reference model, other miners, adversarial"],
        ["Baseline", router.get("reference") or "reference router, not published yet"],
    ]


def _escalation_rows(escalation: dict[str, Any]) -> list[list[str]]:
    models = " · ".join(f"{m['version']} ({_price(m)})" for m in escalation["frontier"])
    if escalation.get("rows"):
        rows = [list(map(str, row)) for row in escalation["rows"]]
        return [*rows[:2], ["Frontier models", models], *rows[2:]]
    return [
        ["Endpoint", "POST /escalation/repair"],
        ["Input", "task, draft, errors, trace, ticket"],
        ["Output", "repaired answer in the task output format"],
        ["Frontier access", "relay only, with the ticket in the x-mt-ticket header"],
        ["Frontier models", models],
        ["Spend cap", f"{escalation['spend_cap_multiple']:g} × frontier redo cost per item"],
        ["Baseline", "frontier redo pass rate and cost"],
    ]


def _router_score(kind: str, floors: list[float], latency: int) -> list[dict[str, str]]:
    paid = {
        "name": "Paid when",
        "formula": (
            f"≥ {MIN_ITEMS} drafts and the 5th percentile of S_r − S_ref over "
            f"{RESAMPLES} family resamples is above 0"
        ),
    }
    score = {
        "name": "Score",
        "formula": (
            f"S_r = {ROUTER_MEAN_WEIGHT}·mean(s) + {ROUTER_MIN_WEIGHT}·min(s) "
            f"if p95 latency < {_int(latency)} ms, otherwise 0"
        ),
    }
    if kind == "brier_auroc":
        return [
            {
                "name": "Per source",
                "note": "Brier_base is the Brier score of always predicting the share of passing drafts.",
                "formula": "s = 0.5·max(0, 1 − Brier / Brier_base) + 0.5·max(0, 2·AUROC − 1)",
            },
            score,
            paid,
        ]
    return [
        {
            "name": "Loss per draft",
            "formula": (
                "ℓ(p, y) = (1/F)·Σ_f [ f·1(p ≥ f, y = 0) + (1 − f)·1(p < f, y = 1) ] "
                f"+ {BRIER_WEIGHT}·(p − y)²,  f ∈ {{{_floors(floors)}}}"
            ),
        },
        {"name": "Per source", "formula": "s = max(0, 1 − mean ℓ(p, y) / mean ℓ(ȳ, y))"},
        score,
        paid,
    ]


def _grader(block: dict[str, Any]) -> str:
    text = GRADERS[block["grader_type"]]
    probe = block.get("probe")
    if block["grader_type"] == "model_assisted" and probe:
        text += (
            f", probe `{probe['model']}` {probe['file']} at `{probe['revision'][:12]}`, "
            f"{probe['max_tokens']} tokens, on the validator's CPU"
        )
    return text


def _window(timing: dict[str, Any]) -> str:
    margin = int(timing["settle_margin"])
    return (
        "Freeze 1 at the open block · inputs released when reveals close · transcripts "
        f"uploaded before Freeze 2 at the close block · grading in the {_int(margin)} block "
        f"settle margin (about {margin / BLOCKS_PER_HOUR:g} h) before settlement"
    )


def _evaluation(arena: dict[str, Any]) -> dict[str, Any]:
    block = arena["evaluation"]
    rows: list[list[str]] = [["Mode", MODES[block["mode"]]]]
    if "grader_type" in block:
        rows.append(["Grader", _grader(block)])
    if block["mode"] == "miner":
        rows.append(["Window", _window(arena["timing"])])
    if "spot_check" in block:
        spot = block["spot_check"]
        rows.append(
            [
                "Spot check",
                (
                    f"at least {_int(spot['min_calls'])} SLM calls per sampled miner, exact token "
                    f"match on the miner's profile; about {spot['validators_per_miner']} "
                    f"validators per miner; {spot['reference_calls']} reference calls timed on "
                    "the same card"
                ),
            ]
        )
    if "ban" in block:
        ban = block["ban"]
        rows.append(
            [
                "Ban",
                (
                    f"score 0 and a hotkey ban for {ban['epochs']} epochs once one piece of "
                    f"evidence has {ban['confirmations']} confirmations from distinct validators"
                ),
            ]
        )
    if "canaries" in block:
        rows.append(
            [
                "Canaries",
                (
                    f"{block['canaries']['count']} fabricated facts per epoch; a model that "
                    "completes an earlier epoch's canary is banned like a spot check failure"
                ),
            ]
        )
    if "relay_allowance_micros" in block:
        rows.append(
            ["Relay allowance", f"${block['relay_allowance_micros'] / 1e6:g} per miner per epoch"]
        )
    if "computed_cost" in block:
        cost = block["computed_cost"]
        weights = ", ".join(f"{k} {v:g}" for k, v in cost["bytes_per_weight"].items())
        cap = cost["cap_units_per_item"]
        rows.append(
            [
                "Computed cost",
                f"bytes per weight {weights}; "
                + (f"cap {_int(cap)} units per item" if cap else "no cap per item"),
            ]
        )
    profiles = [
        [
            name,
            entry["name"],
            ", ".join(entry.get("gpus") or ["any x86 CPU"])
            + (f" (joining: {', '.join(entry['joining'])})" if entry.get("joining") else ""),
            entry["precision"],
            entry["batching"],
            entry["status"].replace("_", " "),
        ]
        for name, entry in (arena.get("profiles") or {}).items()
    ]
    images = [
        [name, f"`{entry['image']}`", f"`{entry['digest']}`"]
        for name, entry in (block.get("images") or {}).items()
    ]
    return {"mode": block["mode"], "rows": rows, "profiles": profiles, "images": images}


def build(arena: dict[str, Any]) -> dict[str, Any]:
    slm = arena["slm"]
    envelope = arena["envelope"]
    harness = arena["harness"]
    router = arena["router"]
    escalation = arena["escalation"]
    scoring = arena["scoring"]
    floors = arena["floors"]
    limits_hold = (
        " if the harness limits hold, otherwise 0" if harness.get("max_model_calls") else ""
    )
    empty = (
        "Empty pools burn."
        if scoring["empty_pools"] == "burn"
        else "An empty pool's share goes to the winners of the other pools."
    )
    page = {
        "id": arena["id"],
        "title": arena["title"],
        "category": arena["category"],
        "network": arena["network"],
        "purpose": arena["purpose"],
        "slm": {
            "bases": [
                {
                    "model": b["model"],
                    "licence": b["licence_name"],
                    "licence_url": b["licence_url"],
                    "condition": b.get("condition", ""),
                }
                for b in slm["bases"]
            ],
            "reference": slm["reference"],
            "dataset": slm.get("dataset"),
            "evaluation": [*slm["evaluation"], *mechanism(scoring, envelope)],
        },
        "envelope": {
            "group": envelope["group"],
            "name": envelope["name"],
            "docs": envelope["docs"],
            "overrides": [
                limit_row(key, value, envelope["device"])
                for key, value in envelope["overrides"].items()
            ],
        },
        "harness": {
            "rows": _harness_rows(harness),
            "score": [
                {
                    "name": "Score",
                    "formula": (
                        f"S_h = max(0, LB_paired(H − B*)) · min(1, {TOKEN_ALLOWANCE} · tokens_B* / tokens_H)"
                        f"{limits_hold}"
                    ),
                },
                {
                    "name": "Where",
                    "formula": (
                        "B* is the best baseline; LB_paired is the one sided 95% lower bound of the "
                        "per item pass difference, clustered by family"
                    ),
                },
                {"name": "Paid when", "formula": f"n ≥ {scoring['min_items']} and S_h > 0"},
            ],
        },
        "router": {
            "rows": _router_rows(router),
            "score": _router_score(scoring["router"], floors, router["latency_gate_ms"]),
        },
        "escalation": {
            "rows": _escalation_rows(escalation),
            "score": [
                {
                    "name": "Gate",
                    "formula": f"LB(repair passes) ≥ redo pass rate − {ESCALATION_MARGIN}",
                },
                {
                    "name": "Score",
                    "formula": "S_e = max(0, 1 − repair_cost / redo_cost) if the gate holds, otherwise 0",
                },
                {
                    "name": "Paid when",
                    "formula": (
                        f"n ≥ {scoring['min_items']} and the 5th percentile of S_e − S_ref over "
                        f"{RESAMPLES} family resamples is above 0"
                    ),
                },
            ],
        },
        "system": {
            "rows": [
                ["Floors", _floors(floors)],
                ["Reference", "frontier answer cost and p95 latency"],
            ],
            "score": [
                {
                    "name": "Per floor",
                    "formula": (
                        f"v_f = clamp(0, ln 10, {COST_WEIGHT}·ln(C_ref/C_f) + {LATENCY_WEIGHT}·ln(L_ref/L_f)) "
                        "/ ln 10 if LB(pass at f) ≥ f, otherwise 0"
                    ),
                },
                {"name": "Score", "formula": "S_sys = mean of v_f over the floors"},
                {"name": "Paid when", "formula": f"n ≥ {scoring['min_items']} and S_sys > 0"},
            ],
        },
        "scoring": {
            "rows": [[name, f"{POOL_SHARE * 100:g}%"] for name in COMPONENTS],
            "split": (
                "Within each component the top five eligible miners get "
                f"{', '.join(f'{p}%' for p in RANK_PAYOUT[:-1])} and {RANK_PAYOUT[-1]}%. "
                f"Published scores move on a {LADDER_STEP:g} ladder. {empty}"
            ),
        },
        "item_counts": arena["item_counts"],
    }
    if arena.get("evaluation"):
        page["evaluation"] = _evaluation(arena)
    return page


def _escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _table(head: tuple[str, ...], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _cell(value: str) -> str:
    if len(value) <= 40 or " · " not in value:
        return value
    return "<br>".join(value.split(" · "))


def _across(rows: list[list[str]], most: int = 0) -> list[str]:
    count = -(-len(rows) // most) if most else 1
    size = -(-len(rows) // count)
    lines: list[str] = []
    for start in range(0, len(rows), size):
        band = rows[start : start + size]
        if lines:
            lines.append("")
        lines += [
            "| " + " | ".join(name for name, _ in band) + " |",
            "|" + "---|" * len(band),
            "| " + " | ".join(_cell(value) for _, value in band) + " |",
        ]
    return lines


def _formulas(items: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.append(f"*{item['name']}*")
        if item.get("note"):
            lines += ["", item["note"]]
        lines += ["```", item["formula"], "```", ""]
    return lines


def _dataset(dataset: dict[str, Any] | None) -> list[str]:
    if not dataset:
        return ["Not attached yet."]
    lines = [f"[{dataset['name']}]({dataset['url']})"]
    if dataset.get("licence"):
        lines[0] += f" · {dataset['licence']}"
    if dataset.get("about"):
        lines += ["", _escape(dataset["about"])]
    return lines


def markdown(page: dict[str, Any]) -> str:
    purpose = page["purpose"]
    slm = page["slm"]
    reference = slm["reference"]
    envelope = page["envelope"]
    conditions = any(b["condition"] for b in slm["bases"])
    head = ("Model", "Licence", "Condition") if conditions else ("Model", "Licence")
    bases = [
        [f"`{b['model']}`", f"[{b['licence']}]({b['licence_url']})", b["condition"]][: len(head)]
        for b in slm["bases"]
    ]
    lines = [
        f"# {page['title']}",
        "",
        f"Arena ID: `{page['id']}` · Category: {page['category']} · Network: {page['network']}",
        "",
        "## 1. Purpose",
        "",
        _escape(purpose["summary"]),
        "",
        f"**Input:** {_escape(purpose['input'])}",
        "",
        f"**Output:** {_escape(purpose['output'])}",
        "",
        "## 2. Specialised SLM",
        "",
        "**Allowed base models**",
        "",
        *_table(head, bases),
        "",
        f"Reference model: `{reference['model']}` at `{reference['revision'][:12]}`",
        "",
        "**Dataset**",
        "",
        *_dataset(slm.get("dataset")),
        "",
        "**Evaluation criteria**",
        "",
        *_formulas(slm["evaluation"]),
        "## 3. Envelope",
        "",
        f"[{envelope['group']} {envelope['name']}]({envelope['docs']})",
        "",
    ]
    if envelope["overrides"]:
        lines += ["Stricter limits in this arena:", "", *_across(envelope["overrides"], 4), ""]
    lines += [
        "## 4. Harness specification",
        "",
        *_across(page["harness"]["rows"], 4),
        "",
        *_formulas(page["harness"]["score"]),
        "## 5. Router specification",
        "",
        *_across(page["router"]["rows"], 4),
        "",
        *_formulas(page["router"]["score"]),
        "## 6. Escalation repair specification",
        "",
        *_across(page["escalation"]["rows"], 4),
        "",
        *_formulas(page["escalation"]["score"]),
        "## 7. Full system",
        "",
        *_across(page["system"]["rows"], 4),
        "",
        *_formulas(page["system"]["score"]),
        "## 8. Scoring",
        "",
        *_across(page["scoring"]["rows"]),
        "",
        page["scoring"]["split"],
        "",
        "**Items per epoch**",
        "",
        *_across([[test, _int(count)] for test, count in page["item_counts"].items()]),
        "",
    ]
    if page.get("evaluation"):
        lines += _evaluation_section(page["evaluation"])
    return "\n".join(lines)


def _evaluation_section(evaluation: dict[str, Any]) -> list[str]:
    lines = ["## 9. Evaluation and hardware profiles", "", *_across(evaluation["rows"], 2), ""]
    if evaluation["profiles"]:
        lines += [
            "**Allowed hardware profiles**",
            "",
            (
                "A profile is one GPU architecture, one driver and one image digest. A miner "
                "declares its profile in its commitment and is spot checked only by validators "
                "on the same profile."
            ),
            "",
            *_table(
                ("Profile", "Name", "GPUs", "Precision", "Batching", "Status"),
                evaluation["profiles"],
            ),
            "",
        ]
    if evaluation["images"]:
        lines += [
            "**Official images, pulled by digest**",
            "",
            *_table(("Image", "Name", "Digest"), evaluation["images"]),
            "",
        ]
    return lines
