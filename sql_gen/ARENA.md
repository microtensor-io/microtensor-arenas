# Databases: SQL generation

Arena ID: `sql_gen` · Category: Text · Network: testnet

## 1. Purpose

Teams and apps ask questions of their own databases in plain language. This specialist turns each question into one correct SQLite query on one GPU at a fraction of frontier cost.

**Input:** The database schema as CREATE TABLE statements (system message) and the question (user message)

**Output:** JSON {"sql": "..."}: one SQLite SELECT statement

## 2. Specialised SLM

**Allowed base models**

| Model | Licence |
|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE) |
| `microsoft/Phi-4-mini-instruct` | [MIT](https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE) |

Reference model: `Qwen/Qwen2.5-Coder-7B-Instruct` at `c03e6d358207`

**Dataset**

Not attached yet.

**Evaluation criteria**

*Item pass*
```
pass(i) = 1 if the query's result matches the reference query's result on every database state of the item, otherwise 0
```

*Wilson lower bound on the pass rate (n items, pass rate p, z = 1.645)*
```
LB = (p + z²/2n − z·√(p(1 − p)/n + z²/4n²)) / (1 + z²/n)
```

*Twin factor (gap is the pass rate difference between original items and their twins)*
```
twin = 1 − max(0, gap − 0.03)
```

*Gate*
```
G = 1 if TTFT p95 ≤ 2,000 ms, TPOT p95 ≤ 60 ms, peak memory ≤ 24 GB and the weights pass provenance against the declared base, otherwise 0
```

*Efficiency against the reference model*
```
E = clamp(−1, 1, 0.4·ln(TTFT_ref/TTFT) + 0.3·ln(mem_ref/mem) + 0.2·ln(N_ref/N) + 0.1·ln(TPOT_ref/TPOT))
```

*SLM score*
```
S_slm = clamp(0, 1, G · LB · twin · (1 + 0.25·E))
```

*Paid when*
```
n ≥ 400 and the paired lower bound of (miner − reference model) per item, clustered by family, is above 0
```

## 3. Envelope

[mt-5 Small GPU](https://microtensor.cloud/docs/certified-inference/system/envelope-groups)

Stricter limits in this arena:

| First token (p95) | Max output | Max input |
|---|---|---|
| ≤ 2,000 ms | 256 tokens | 32,512 tokens |

## 4. Harness specification

| Endpoint | Input | Output |
|---|---|---|
| POST /harness/run | {"item_id", "task"} with the task fields from the input schema | {"answer": ...} in the task output format |

| Tools | Sandbox | Request timeout |
|---|---|---|
| schema_lookup, query_execution | gVisor, 4 CPU, 8 GB, no network except the model, router, tools and relay | 120 s |

| Code size | Limits | Baselines |
|---|---|---|
| ≤ 2 MB with escalation | 32 steps, 200,000 tokens | bare model under the arena prompt, reference harness and its variants |

*Score*
```
S_h = max(0, LB_paired(H − B*)) · min(1, 1.5 · tokens_B* / tokens_H)
```

*Where*
```
B* is the best baseline; LB_paired is the one sided 95% lower bound of the per item pass difference, clustered by family
```

*Paid when*
```
n ≥ 400 and S_h > 0
```

## 5. Router specification

| Model | Max size | Max input | Request |
|---|---|---|---|
| sequence classifier with one output | ≤ 100,000,000 parameters | 8,192 tokens | POST /decide {"input": {"query", "draft"}} |

| Response | Latency (p95) | Drafts | Baseline |
|---|---|---|---|
| {"answers": [{"label": "draft_correct", "probability": p}]} | < 100 ms | your model, the reference model, other miners, adversarial | reference router, not published yet |

*Loss per draft*
```
ℓ(p, y) = (1/F)·Σ_f [ f·1(p ≥ f, y = 0) + (1 − f)·1(p < f, y = 1) ] + 0.2·(p − y)²,  f ∈ {0.8 · 0.9 · 0.95 · 0.99}
```

*Per source*
```
s = max(0, 1 − mean ℓ(p, y) / mean ℓ(ȳ, y))
```

*Score*
```
S_r = 0.7·mean(s) + 0.3·min(s) if p95 latency < 100 ms, otherwise 0
```

*Paid when*
```
≥ 400 drafts and the 5th percentile of S_r − S_ref over 200 family resamples is above 0
```

## 6. Escalation repair specification

| Endpoint | Input | Output | Frontier access |
|---|---|---|---|
| POST /escalation/repair | task, draft, errors, trace, ticket | repaired answer in the task output format | relay only, with the ticket in the x-mt-ticket header |

| Frontier models | Spend cap | Baseline |
|---|---|---|
| gpt-5-2026-06-01 ($1.25 in, $10 out per M tokens)<br>claude-sonnet-5-5-20260601 ($3 in, $15 out per M tokens) | 1.5 × frontier redo cost per item | frontier redo pass rate and cost |

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
n ≥ 400 and the 5th percentile of S_e − S_ref over 200 family resamples is above 0
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
n ≥ 400 and S_sys > 0
```

## 8. Scoring

| Specialised SLM | Harness | Router | Escalation repair | Full system |
|---|---|---|---|---|
| 20% | 20% | 20% | 20% | 20% |

Within each component the top five eligible miners get 40%, 25%, 15%, 12% and 8%. Published scores move on a 0.01 ladder. An empty pool's share goes to the winners of the other pools.

**Items per epoch**

| slm | harness | router | escalation | system |
|---|---|---|---|---|
| 400 | 400 | 250 | 120 | 400 |

## 9. Evaluation and hardware profiles

| Mode | Grader |
|---|---|
| miner: each miner runs its own evaluation on its own GPU and uploads a transcript; validators replay it on CPU and spot check sampled calls on the same hardware profile | pure CPU |

| Window | Spot check |
|---|---|
| Freeze 1 at the open block<br>inputs released when reveals close<br>transcripts uploaded before Freeze 2 at the close block<br>grading in the 1,800 block settle margin (about 6 h) before settlement | at least 200 SLM calls per sampled miner, exact token match on the miner's profile; about 2 validators per miner; 20 reference calls timed on the same card |

| Ban | Canaries |
|---|---|
| score 0 and a hotkey ban for 3 epochs once one piece of evidence has 2 confirmations from distinct validators | 3 fabricated facts per epoch; a model that completes an earlier epoch's canary is banned like a spot check failure |

| Relay allowance | Computed cost |
|---|---|
| $2 per miner per epoch | bytes per weight bf16 2, fp16 2, awq 0.5, gptq 0.5, gguf_q4 0.5, gguf_q8 1; no cap per item |

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
