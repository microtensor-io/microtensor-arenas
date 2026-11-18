# Databases: SQL generation

Arena ID: `sql_gen` · Category: Text · Network: testnet

## 1. Purpose

Teams and apps ask questions of their own databases in plain language. This specialist turns each question into one correct SQLite query on one GPU at a fraction of frontier cost.

**Input:** The database schema as CREATE TABLE statements (system message) and the question (user message)

**Output:** JSON {"sql": "..."}: one SQLite SELECT statement

## 2. Specialised SLM

**Base model**

| Model | Licence |
|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE) |

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
n ≥ 25 and the paired lower bound of (miner − reference model) per item, clustered by family, is above 0
```

## 3. Envelope

[mt-5 Small GPU](https://microtensor.cloud/docs/certified-inference/system/envelope-groups)

Stricter limits in this arena:

| First token | Max output | Max input |
|---|---|---|
| ≤ 2,000 ms | 256 tokens | 32,512 tokens |

## 4. Harness specification

| Endpoint | Input | Output | Tools |
|---|---|---|---|
| POST /harness/run | {"item_id", "task"} with the task fields from the input schema | {"answer": ...} in the task output format | schema_lookup, query_execution |

| Sandbox | Request timeout | Code size | Limits |
|---|---|---|---|
| gVisor, 4 CPU, 8 GB, no network except the model, router, tools and relay | 120 s | ≤ 2 MB with escalation | 32 steps, 200,000 tokens |

No pool of its own: the harness is scored inside the full system (section 7).

## 5. Router specification

| Model | Max size | Max input |
|---|---|---|
| sequence classifier with one output | ≤ 100,000,000 parameters | 8,192 tokens |

| Request | Response |
|---|---|
| POST /decide {"input": {"query", "draft"}} | {"answers": [{"label": "draft_correct", "probability": p}]} |

No pool of its own: the router is scored inside the full system (section 7), where a router over its size limit or copied from an earlier submission scores 0.

## 6. Escalation repair specification

| Endpoint | Input | Output |
|---|---|---|
| POST /escalation/repair | task, draft, errors, trace, ticket | repaired answer in the task output format |

| Frontier access | Frontier models | Spend cap |
|---|---|---|
| relay only, with the ticket in the x-mt-ticket header | gpt-5-2026-06-01 ($1.25 in, $10 out per M tokens)<br>claude-sonnet-5-5-20260601 ($3 in, $15 out per M tokens) | 1.5 × frontier redo cost per item |

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
| on the profile card all 200 sampled calls run one at a time and 10 of them, chosen by a hash of the call, are timed; every timed call must be within the limits for its input length: first token ≤ 2 s; per token ≤ 60 ms; a call over a limit is rerun once and only its second timing counts; peak memory ≤ 24 GB at full context; a miner over any of these fails the envelope | a mismatch is rerun on the same card after the engine restarts; if both runs agree with each other and differ from the miner, it goes in the coordinator's signed report as evidence; when a validator's own check fails a miner, the miner scores 0 in that validator's report only, with no ban |

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
