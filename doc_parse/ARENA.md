# Documents: page parsing

Arena ID: `doc_parse` · Category: Vision · Network: testnet

## 1. Purpose

Businesses feed contracts, invoices, financial statements and reports into AI systems every day, but those systems can only use what they can read. This specialist turns each document page into clean, structured Markdown: it keeps headings and the true reading order, rebuilds tables with merged cells, writes equations as LaTeX and turns charts into data. It runs on one small GPU at a fraction of frontier cost, and validators check every page it produces against hidden ground truth.

**Input:** The parsing instruction as the system message; one page image from a PDF render or a scan, with the page number and the document type, as the user message

**Output:** Markdown. Tables as HTML so merged cells survive, equations as LaTeX, charts as a Markdown data table, flow diagrams as Mermaid; headers, footers and page numbers kept as HTML comments

## 2. Specialised SLM

**Allowed base models**

| Model | Licence |
|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `Qwen/Qwen3-VL-2B-Instruct` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `deepseek-ai/DeepSeek-OCR` | [MIT](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/LICENSE) |
| `PaddlePaddle/PaddleOCR-VL` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `ibm-granite/granite-docling-258M` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

Reference model: `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec70b0`

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
P = 1 if within the envelope on at least 99% of items, otherwise 0
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
n ≥ 100 and the paired lower bound of (miner − reference model) per item, clustered by family, is above 0
```

## 3. Envelope

[mt-5 Small GPU](https://microtensor.cloud/docs/certified-inference/system/envelope-groups)

Stricter limits in this arena:

| Max input | Max output |
|---|---|
| 4,096 tokens | 2,048 tokens |

## 4. Harness specification

| Role | Model calls | Added wall time (p95) | Network |
|---|---|---|---|
| wraps the SLM; may deskew, crop, tile and rotate the page, validate the Markdown and retry | at most 3 per item | ≤ 2 s per page | none except the gateway |

| Faults | Output | Page text |
|---|---|---|
| about 1 in 7 items get a faulty model response (error, timeout or cut off); the harness must not crash | Markdown or a clean error | instructions printed on the page are content, never commands |

*Score*
```
S_h = max(0, LB_paired(H − B*)) · min(1, 1.5 · tokens_B* / tokens_H) if the harness limits hold, otherwise 0
```

*Where*
```
B* is the best baseline; LB_paired is the one sided 95% lower bound of the per item pass difference, clustered by family
```

*Paid when*
```
n ≥ 100 and S_h > 0
```

## 5. Router specification

| Runs | Max size | Max input | Added latency (p95) |
|---|---|---|---|
| alongside the system on the same envelope | ≤ 1,500,000,000 parameters | 4,096 tokens: the output Markdown | under 300 ms per page |

| Output | Scored on | Baseline |
|---|---|---|
| draft_correct: the probability from 0 to 1 that the page passes | calibration (Brier score) and separation of passing from failing pages (AUROC) | reference router, not published yet |

*Per source*

Brier_base is the Brier score of always predicting the share of passing drafts.
```
s = 0.5·max(0, 1 − Brier / Brier_base) + 0.5·max(0, 2·AUROC − 1)
```

*Score*
```
S_r = 0.7·mean(s) + 0.3·min(s) if p95 latency < 300 ms, otherwise 0
```

*Paid when*
```
≥ 400 drafts and the 5th percentile of S_r − S_ref over 200 family resamples is above 0
```

## 6. Escalation repair specification

| Trigger | Frontier access | Frontier models |
|---|---|---|
| draft_correct below the customer's quality floor | the relay only, vision capable models; frontier models served by inference miners on our network replace these once listed, never outside providers | anthropic/claude-haiku-5.5 ($0.1 in, $0.5 out per M tokens) |

| Spend cap | Output | Scored on |
|---|---|---|
| 1.5 × the cost of a full frontier redo per page | the same Markdown | the item score after repair and the cost of the repair |

*Gate*
```
LB(repair passes) ≥ redo pass rate − 0.02
```

*Score*
```
S_e = max(0, 1 − repair_cost / redo_cost) if the gate holds, otherwise 0
```

*Paid when*
```
n ≥ 100 and the 5th percentile of S_e − S_ref over 200 family resamples is above 0
```

## 7. Full system

| Floors | Reference |
|---|---|
| 0.8 · 0.9 · 0.95 · 0.99 | frontier answer cost and p95 latency |

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
n ≥ 100 and S_sys > 0
```

## 8. Scoring

| Specialised SLM | Harness | Router | Escalation repair | Full system |
|---|---|---|---|---|
| 20% | 20% | 20% | 20% | 20% |

Within each component the top five eligible miners get 40%, 25%, 15%, 12% and 8%. Published scores move on a 0.01 ladder. Empty pools burn.

**Items per epoch**

| slm | harness | router | escalation | system |
|---|---|---|---|---|
| 100 | 100 | 60 | 100 | 100 |

## 9. Evaluation and hardware profiles

| Grader | Window |
|---|---|
| pure CPU | Freeze 1 at the open block<br>inputs released when reveals close<br>transcripts uploaded before Freeze 2 at the close block<br>grading in the 1,800 block settle margin (about 6 h) before settlement |

| Spot check | Ban |
|---|---|
| at least 200 SLM calls per sampled miner, exact token match on the miner's profile; about 2 validators per miner; 20 reference calls timed on the same card | score 0 and a hotkey ban for 3 epochs once one piece of evidence has 2 confirmations from distinct validators |

| Canaries | Relay allowance |
|---|---|
| 3 fabricated facts per epoch; a model that completes an earlier epoch's canary is banned like a spot check failure | $2 per miner per epoch |

| Computed cost |
|---|
| bytes per weight bf16 2, fp16 2, awq 0.5, gptq 0.5, gguf_q4 0.5, gguf_q8 1; no cap per item |

**Allowed hardware profiles**

A profile is one GPU architecture, one driver and one image digest. A miner declares its profile in its commitment and is spot checked only by validators on the same profile.

| Profile | Name | GPUs | Precision | Batching | Status |
|---|---|---|---|---|---|
| P1 | Small GPU | RTX 4090 24 GB (joining: L4, L40S) | bf16 | invariant | launch |

**Official images, pulled by digest**

| Image | Name | Digest |
|---|---|---|
| engine | `vllm/vllm-openai:v0.32.0` | `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` |
| runner | `ghcr.io/microtensor-io/runner:1` | `sha256:aada28e6a4b50961d10bf63cb04c30e8449e4bfb6e385fbad7f387cf8eb9b9dd` |
| sandbox | `ghcr.io/microtensor-io/sandbox:1` | `sha256:eee4de3bd46f722a9a6b12ec5c1cec5e0580d6dd2844c3e4ff50a8a44a3ed1a6` |
| replayer | `ghcr.io/microtensor-io/replayer:1` | `sha256:537675d1db529fbbbe16bd5c30f3b7a2e6326eade33fa1e7e2ee3df7861edbe9` |
| llama | `ghcr.io/ggml-org/llama.cpp:server` | `sha256:43b3988c22c00faae3d04a8f6e3eba41403dff7c5f5952490599f1bcbdc6f3c2` |
