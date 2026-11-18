# Documents: page parsing

Arena ID: `doc_parse` · Category: Vision · Network: testnet

## 1. Purpose

Businesses feed contracts, invoices, financial statements and reports into AI systems every day, but those systems can only use what they can read. This specialist turns each document page into clean, structured Markdown: it keeps headings and the true reading order, rebuilds tables with merged cells, writes equations as LaTeX and turns charts into data. It runs on one small GPU at a fraction of frontier cost, and validators check every page it produces against hidden ground truth.

**Input:** The parsing instruction as the system message; one page image from a PDF render or a scan, with the page number and the document type, as the user message

**Output:** Markdown. Tables as HTML so merged cells survive, equations as LaTeX, charts as a Markdown data table, flow diagrams as Mermaid; headers, footers and page numbers kept as HTML comments

## 2. Specialised SLM

**Base model**

| Model | Licence |
|---|---|
| `lightonai/LightOnOCR-2-1B` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

Reference model: `lightonai/LightOnOCR-2-1B` at `be7c41313c30`

**Dataset**

[allenai/olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) · ODC-BY

Dev split only; hidden items never go in the repo. About 270,000 PDF pages with Markdown answers and HTML tables, written by GPT-4.1, so they are not perfect. Also allowed for training: DocLayNet v1.1 (80,863 pages labelled into 11 layout types, CDLA Permissive 1.0), olmOCR-Bench (1,403 PDFs with 7,010 checks, ODC-BY, a public practice benchmark), PubTabNet (CDLA Permissive 1.0), and VRDU, CORD v2 and FATURA (CC BY 4.0) for forms, receipts and invoices. FinTabNet's licence is still to confirm. OmniDocBench is research only: cite it, never train or grade on it.

**Evaluation criteria**

*Valid output*
```
valid(i) = 1 if the Markdown parses and every HTML table is well formed, otherwise 0
```

*Text*

Body text against the page's text, as one minus the normalised edit distance.
```
text(i) = 1 − edit_distance(output_text, page_text) / max(len(output_text), len(page_text))
```

*Reading order*

The order of every block on the page, including captions, footnotes and two column text, as Kendall's tau between the output order and the true order, not just pairs of lines.
```
order(i) = (τ + 1) / 2
```

*Tables*

Full structure and content against the real table, including merged cells and multi level headers: the TEDS score, averaged over the page's tables.
```
tables(i) = mean TEDS over the page's tables
```

*Equations*

Each equation's LaTeX matches the page after normalisation.
```
equations(i) = matching_equations / equations_on_page
```

*Charts*

The values pulled out of a chart match the data used to draw it, within 2%.
```
charts(i) = matching_values / values_in_chart
```

*Hallucination*

Text that is not on the page, and any text inside regions that should be empty.
```
halluc(i) = invented_characters / output_characters
```

*Item score*

Only the checks that apply to the page count, with weights text 0.35, order 0.15, tables 0.25, equations 0.15, charts 0.10, renormalised over the checks present.
```
item_score(i) = valid(i) · (1 − halluc(i)) · Σ w_k · check_k(i) / Σ w_k
pass(i) = 1 if item_score(i) ≥ 0.85
```

*Wilson lower bound on the pass rate (n items, pass rate p, z = 1.96)*
```
LB = (p + z²/2n − z·√(p(1 − p)/n + z²/4n²)) / (1 + z²/n)
```

*Twin factor (gap is the pass rate difference between original items and their twins)*
```
twin = 1 − min(1, max(0, gap − 0.02) / 0.1)
```

*Provenance gate*
```
G = 1 if the fingerprint matches the declared base and the system beats that base, otherwise 0
```

*Envelope pass*
```
P = 1 if every call the spot check times is within the limits for its input length and peak memory is within the envelope, otherwise 0
```

*Efficiency (ref is the reference baseline, x is TTFT, memory, parameters or TPOT)*
```
e_x = min(1, max(0, (ref − miner) / ref))
E = 0.4·e_TTFT + 0.3·e_mem + 0.2·e_N + 0.1·e_TPOT
```

*Edge bonus*
```
B = 0.1 for each lower envelope group passed, max 0.2
```

*SLM score*
```
S_slm = G · P · LB · twin · (1 + 0.25·E + B)
```

*Paid when*
```
n ≥ 25 and the paired lower bound of (miner − reference model) per item, clustered by family, is above 0
```

## 3. Envelope

[mt-5 Small GPU](https://microtensor.cloud/docs/certified-inference/system/envelope-groups)

Stricter limits in this arena:

| Max input | Max output |
|---|---|
| 4,096 tokens | 2,048 tokens |

## 4. Harness specification

| Role | Model calls | Network |
|---|---|---|
| wraps the SLM; may deskew, crop, tile and rotate the page, validate the Markdown and retry | at most 3 per item | none except the gateway |

| Faults | Output | Page text |
|---|---|---|
| about 1 in 7 items get a faulty model response (error, timeout or cut off); the harness must not crash | Markdown or a clean error | instructions printed on the page are content, never commands |

No pool of its own: the harness is scored inside the full system (section 7).

## 5. Router specification

| Runs | Max size | Max input | Output |
|---|---|---|---|
| alongside the system on the same envelope | ≤ 1,500,000,000 parameters | 4,096 tokens: the output Markdown | draft_correct: the probability from 0 to 1 that the page passes |

No pool of its own: the router is scored inside the full system (section 7), where a router over its size limit or copied from an earlier submission scores 0.

## 6. Escalation repair specification

| Trigger | Frontier access | Frontier models |
|---|---|---|
| draft_correct below the customer's quality floor | the relay only, vision capable models; frontier models served by inference miners on our network replace these once listed, never outside providers | anthropic/claude-haiku-5.5 ($0.1 in, $0.5 out per M tokens) |

| Spend cap | Output |
|---|---|
| 1.5 × the cost of a full frontier redo per page | the same Markdown |

No pool of its own: escalation repair is scored inside the full system (section 7), through the pass rate and cost of the answers.

## 7. Full system

| Pipeline | Floors | Reference |
|---|---|---|
| the specialist model, harness, router and escalation repair run together on every item; parts the miner does not ship come from the reference | 0.8 · 0.9 · 0.95 · 0.99 | frontier answer cost and p95 latency |

*Per floor*
```
v_f = clamp(0, ln 10, 0.7·ln(C_ref/C_f) + 0.3·ln(L_ref/L_f)) / ln 10 if LB(pass at f) ≥ f, otherwise 0
```

*Score*
```
S_sys = mean of v_f over the floors
```

*Paid when*
```
n ≥ 25 and S_sys > 0
```

## 8. Scoring

| Specialist model | Full system |
|---|---|
| 60% | 40% |

The two pools share all of the subnet's miner emission; no share goes to serving or compute. Within each pool the top five eligible miners get 40%, 25%, 15%, 12% and 8%. Published scores move on a 0.01 ladder. Empty pools burn. The harness, router and escalation repair have no pools of their own and count through the full system.

**Items per epoch**

| slm | system |
|---|---|
| 25 | 25 |

## 9. Evaluation and hardware profiles

| Grader | Window |
|---|---|
| pure CPU | Freeze 1 at the open block<br>inputs released after the mirror pass<br>transcripts uploaded before Freeze 2 at the close block<br>grading in the 1,800 block settle margin (about 6 h) before settlement |

| Uploads | Spot check |
|---|---|
| at most 3 per hotkey per epoch; a new upload replaces the last one and gets a new sample | the coordinator, run by the subnet owner, reruns 200 SLM calls per miner on the miner's profile and requires an exact token match; a validator with a GPU of that profile may rerun the same calls with the same check and use its own result; the calls are drawn from those that did not fault, seeded by the transcript hash and the block hash 10 blocks after the upload; 20 reference calls are timed on the same card once per epoch |

| Speed and memory | Evidence |
|---|---|
| on the profile card all 200 sampled calls run one at a time and 10 of them, chosen by a hash of the call, are timed; every timed call must be within the limits for its input length: first token ≤ 4 s; per token ≤ 60 ms; a call over a limit is rerun once and only its second timing counts; peak memory ≤ 24 GB at full context; a miner over any of these fails the envelope | a mismatch is rerun on the same card after the engine restarts; if both runs agree with each other and differ from the miner, it goes in the coordinator's signed report as evidence; when a validator's own check fails a miner, the miner scores 0 in that validator's report only, with no ban |

| No result | Ban |
|---|---|
| a miner with no spot check result for its upload scores 0 in the specialist model and full system pools for that epoch; it is not a ban and there is no appeal; a check that ran is a result even when it reproduced few calls; when no coordinator report reaches the server by 150 blocks before settlement, validators keep the last settled weights and the epoch settles void, so the arena keeps paying the previous settlement | on standing evidence validators score the miner 0 for that epoch and apply a 3 epoch hotkey ban, counting that epoch; the server never bans directly |

| Appeal | Duplicates |
|---|---|
| within one epoch the miner can appeal; the flagged call is rerun and a match lifts the ban | every validator compares each model and router with earlier submissions before scoring, whichever hotkey or coldkey made them; the earliest commit on chain wins, the earlier transaction when two share a block, and a sealed commitment counts from its sealed commit, not its reveal; a duplicated model scores 0 in the specialist model and full system pools and a duplicated router in the full system pool for that epoch; when only the router is a duplicate the model is still scored |

| Canaries | Relay allowance |
|---|---|
| 3 fabricated facts per epoch; a model that completes an earlier epoch's canary is banned like a spot check failure | $2 per miner per epoch |

| Computed cost |
|---|
| bytes per weight bf16 2, fp16 2, awq 0.5, gptq 0.5, gguf_q4 0.5, gguf_q8 1; no cap per item |

**Allowed hardware profiles**

A profile is one GPU architecture, one driver and one image digest. A miner declares its profile in its commitment and is spot checked on the same profile, by the coordinator and by any validator with a GPU of that profile. An arena lists only the profiles the coordinator checks, because a miner with no spot check result scores 0.

| Profile | Name | GPUs | Precision | Batching | Status |
|---|---|---|---|---|---|
| P1 | Small GPU | RTX 4090 24 GB (joining: L4, L40S) | bf16 | single | launch |

**Official images, pulled by digest**

| Image | Name | Digest |
|---|---|---|
| engine | `vllm/vllm-openai:v0.32.0` | `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` |
| runner | `ghcr.io/microtensor-io/runner:1` | `sha256:aada28e6a4b50961d10bf63cb04c30e8449e4bfb6e385fbad7f387cf8eb9b9dd` |
| sandbox | `ghcr.io/microtensor-io/sandbox:1` | `sha256:eee4de3bd46f722a9a6b12ec5c1cec5e0580d6dd2844c3e4ff50a8a44a3ed1a6` |
| replayer | `ghcr.io/microtensor-io/replayer:1` | `sha256:537675d1db529fbbbe16bd5c30f3b7a2e6326eade33fa1e7e2ee3df7861edbe9` |
| llama | `ghcr.io/ggml-org/llama.cpp:server` | `sha256:43b3988c22c00faae3d04a8f6e3eba41403dff7c5f5952490599f1bcbdc6f3c2` |
