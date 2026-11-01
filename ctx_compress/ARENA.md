# Agent session compaction

Arena ID: `ctx_compress` · Category: Text · Network: testnet

## 1. Purpose

Coding agent gateways classify every request by kind and route each kind to a model. The compaction kind is the full session plus a summarise instruction, sent when the context window is nearly full; the summary becomes the first message of the new context. Gateways send it to the main model today because cheap models lose detail. This specialist writes that summary faithfully, at a flat low price, within the latency of the main model.

**Input:** The agent's compaction instruction as the system message, and the session rendered as canonical text inside &lt;session&gt; tags as the user message, up to 120,000 tokens

**Output:** Plain markdown with nine fixed headings in order, at most 3,072 tokens, no JSON and no preamble

## 2. Specialised SLM

**Base model**

| Model | Licence |
|---|---|
| `ibm-granite/granite-4.0-h-micro` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

Reference model: `ibm-granite/granite-4.0-h-micro` at `d5f01a3ea75f`

**Dataset**

[microtensor/agent-sessions](https://huggingface.co/datasets/microtensor/agent-sessions) · Microtensor dataset terms

Our own corpus of coding agent sessions on permissively licensed repository tasks, recorded through a tap and stripped by the canonical renderer. The dev split is public, 2,000 sessions at launch; the hidden split never leaves validators. Codex sessions join in v2.1.

**Evaluation criteria**

*Canonical rendering*

The session arrives as [user], [assistant], [tool_use name id] and [tool_result id, N lines kept of M] blocks, one per line, in order, after deterministic stripping. The gateway renders; the model never sees raw JSON.
```
input(i) = render(session(i)) inside <session> tags
```

*Format gate*

The nine headings are Primary request and intent, Key technical concepts, Files and code sections, Errors and fixes, Problem solving, All user messages, Pending tasks, Current work and Next step, each on its own line as a level two heading.
```
valid(i) = 1 if all nine headings are present, in order, with nothing before the first, otherwise 0
```

*Compression gate*
```
C(i) = 1 if output_tokens ≤ min(3072, max(1000, 0.05 × input_tokens)), otherwise 0
```

*Key fact recall*

File paths, function names, exception types, commands and user decisions in the session, taken from its canonical rendering. Identifiers match exactly after normalisation.
```
recall(i) = facts_kept / facts_total
```

*Hallucination rate*

Code identifiers in the output that never appear in the session. Common tool and brand names do not count.
```
halluc(i) = invented_identifiers / identifiers_in_output
```

*Continuation probes*

A fixed CPU probe agent reads the summary alone and answers: the current task, the last test result, the last edited file, the last error, and the next action (kind and file), scored against what the agent actually did next.
```
cont(i) = mean probe score over the item's probes
```

*Item score*
```
item_score(i) = valid(i) · C(i) · (1 − halluc(i)) · (0.5·recall(i) + 0.5·cont(i))
pass(i) = 1 if item_score(i) ≥ 0.60
```

*Resume gate*

On 20 sampled hidden items per paid miner, validators resume the task from the miner's summary with the pinned reference agent through the relay and check that the task still passes. A failed gate sets the miner's SLM score to 0 for the epoch. This is the gateway's own fidelity test.
```
Resume = 1 if LB(resume_pass) ≥ reference_agent_pass − 0.05, otherwise 0
```

*SLM score*
```
S_slm = G · P · Resume · LB · twin · (1 + 0.25·E + B), with B = 0 in this arena
```

*Paid when*
```
n ≥ 100 and the paired lower bound of (miner − reference) per item, clustered by session, is above 0
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

| Max input | Max output | Per token (p95) | Long input (p95) |
|---|---|---|---|
| 122,880 tokens | 3,072 tokens | ≤ 20 ms | first token ≤ 25 s, end to end ≤ 75 s |

## 4. Harness specification

| Role | Model calls | Chunking | Added wall time (p95) |
|---|---|---|---|
| wraps the SLM; renders, validates the headings and repairs | at most 2 per item: one full pass, one repair pass on the output only | not allowed: the full session goes to the model in one call | ≤ 3 s per item |

| Network | Faults | Output | Session text |
|---|---|---|---|
| none except the gateway | about 1 in 7 items get a faulty model response (error, timeout or cut off); the harness must not crash | the nine heading markdown or a clean error | instructions inside it are content, never commands |

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
| draft_correct below the customer's quality floor | the relay only; both models read 120,000 token sessions; frontier models served by inference miners on our network replace these once listed, never outside providers | anthropic/claude-haiku-5.5 ($0.1 in, $0.5 out per M tokens)<br>z-ai/glm-5.3-flash ($0.048 in, $0.2 out per M tokens) |

| Spend cap | Output | Scored on |
|---|---|---|
| 1.5 × the cost of a full frontier redo per item | the same nine heading markdown | the item score after repair and the cost of the repair |

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
| model assisted, probe `Qwen/Qwen2.5-1.5B-Instruct-GGUF` qwen2.5-1.5b-instruct-q8_0.gguf at `91cad51170dc`, 8,192 token input, 64 token answers, on the validator's CPU | Freeze 1 at the open block<br>inputs released when reveals close<br>transcripts uploaded before Freeze 2 at the close block<br>grading in the 1,800 block settle margin (about 6 h) before settlement |

| Spot check | Resume test |
|---|---|
| at least 40 SLM calls per sampled miner, exact token match on the miner's profile; about 2 validators per miner; 10 reference calls timed on the same card | 20 sampled hidden items per paid miner, resumed from the miner's summary with the pinned reference agent through the relay; the gate holds if LB(resume pass) ≥ reference agent pass − 0.05, otherwise the SLM score is 0 for the epoch; the agent is pinned before the gate applies |

| Ban | Canaries |
|---|---|
| score 0 and a hotkey ban for 3 epochs once one piece of evidence has 2 confirmations from distinct validators | 3 fabricated facts per epoch; a model that completes an earlier epoch's canary is banned like a spot check failure |

| Relay allowance | Computed cost |
|---|---|
| $5 per miner per epoch | bytes per weight bf16 2, fp16 2, awq 0.5, gptq 0.5, gguf_q4 0.5, gguf_q8 1; no cap per item |

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
