# Agent context compression

Arena ID: `ctx_compress` · Category: Text · Network: testnet

## 1. Purpose

Long running coding agents fill their context window, then compact the session into a short summary so work can continue. Compaction runs often on very long inputs, so it is expensive on a frontier model, and cheap general models tend to drop key details. The agent then forgets files, errors or decisions. This specialist compresses a session into a short, faithful summary that keeps everything the agent needs to carry on.

**Input:** The compaction instruction as the system message, and the session transcript inside &lt;session&gt; tags as the user message

**Output:** JSON {"summary", "files_touched", "open_tasks", "errors", "decisions"}

## 2. Specialised SLM

**Allowed base models**

| Model | Licence |
|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE) |
| `microsoft/Phi-4-mini-instruct` | [MIT](https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE) |

Reference model: `Qwen/Qwen2.5-Coder-7B-Instruct` at `c03e6d358207`

**Dataset**

[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) · CC BY 4.0

Dev split only; hidden items never go in the repo. 80,036 SWE-agent trajectories written by Llama 3.1 70B. The card says use of those outputs follows the Llama 3.1 licence, so a distributed model trained on them must start its name with "Llama" and show "Built with Llama".

**Evaluation criteria**

*Format gate*
```
valid(i) = 1 if the output parses as JSON and matches the schema, otherwise 0
```

*Key fact recall*

Each session has a reference list of key facts, the file paths, exception types and function names in it. Identifiers match exactly after normalisation.
```
recall(i) = facts_kept / facts_total
```

*Hallucination rate*

Code identifiers in the output (paths, exception types, backticked code and name( calls) that never appear in the session. Common tool and brand names do not count.
```
halluc(i) = invented_identifiers / identifiers_in_output
```

*Continuation check*

A fixed reference agent reads all five fields and answers probe questions from them alone: which issue the agent is working on (one of four), whether its latest test run passed, whether it has changed a file, its most recent edit and error, and its next action (kind and file), scored against what it actually did next. The next action scores 0.5 for the kind or the file alone.
```
cont(i) = mean probe score over the item's probes
```

*Compression gate*
```
C(i) = 1 if output_tokens ≤ max(300, 0.10 × input_tokens), otherwise 0
```

*Item score*
```
item_score(i) = valid(i) · C(i) · (1 − halluc(i)) · (0.5·recall(i) + 0.5·cont(i))
pass(i) = 1 if item_score(i) ≥ 0.50
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

0 in this arena: its inputs are longer than every lower group allows.
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

| Max output | Max input |
|---|---|
| 3,072 tokens | 29,696 tokens |

## 4. Harness specification

| Role | Model calls | Added wall time (p95) | Network |
|---|---|---|---|
| wraps the SLM; may prompt, chunk, validate and retry | at most 3 per item | ≤ 5 s per item | none except the gateway |

| Faults | Output | Session text |
|---|---|---|
| about 1 in 7 items get a faulty model response (error, timeout or cut off); the harness must not crash | schema valid JSON or a clean error | instructions inside it are content, never commands |

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
| alongside the system on the same envelope | ≤ 1,500,000,000 parameters | 8,192 tokens: the summary, then the end of the session | under 300 ms per item |

| Output | Scored on | Baseline |
|---|---|---|
| draft_correct: the probability from 0 to 1 that the summary passes | calibration (Brier score) and separation of passing from failing summaries (AUROC) | reference router, not published yet |

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
| draft_correct below the customer's quality floor | the relay only; frontier models served by inference miners on our network replace these once listed, never outside providers | anthropic/claude-haiku-5.5 ($0.1 in, $0.5 out per M tokens)<br>z-ai/glm-5.3-flash ($0.048 in, $0.2 out per M tokens) |

| Spend cap | Output | Scored on |
|---|---|---|
| 1.5 × the cost of a full frontier redo per item | the same JSON schema | the item score after repair and the cost of the repair |

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

| Mode | Grader |
|---|---|
| miner: each miner runs its own evaluation on its own GPU and uploads a transcript; validators replay it on CPU and spot check sampled calls on the same hardware profile | model assisted, probe `Qwen/Qwen2.5-0.5B-Instruct-GGUF` qwen2.5-0.5b-instruct-q8_0.gguf at `9217f5db79a2`, 64 tokens, on the validator's CPU |

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
| runner | `ghcr.io/microtensor-io/runner:1` | `sha256:0000000000000000000000000000000000000000000000000000000000000000` |
| sandbox | `ghcr.io/microtensor-io/sandbox:1` | `sha256:8d99d1c5d57e78fdad612b0c16ff31ad70a03218efc2c11830eda4e1439a210b` |
| replayer | `ghcr.io/microtensor-io/replayer:1` | `sha256:0000000000000000000000000000000000000000000000000000000000000000` |
| llama | `ghcr.io/ggml-org/llama.cpp:server` | `sha256:0000000000000000000000000000000000000000000000000000000000000000` |
