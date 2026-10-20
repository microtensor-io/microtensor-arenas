# Changelog

## Season 1

- Replaces session titling (cc_aside_s1). Envelope group mt-3 with max output 3,072 and max input 29,696 tokens, so Qwen2.5's 32,768 token window holds both. Bases Qwen2.5 Coder 7B and 1.5B and Phi 4 mini. Escalation through the relay (Claude Haiku 5.5, GLM 5.3 Flash) until frontier models are served on our network. Empty pools burn.
- Testnet runs on a 100 item hidden set: 100 items each for SLM, harness, escalation and system, 60 for the router, and a pool is paid from 100 items (`scoring.min_items`). Mainnet returns to 400.
- Envelope classes version 2: the Small GPU class is now mt-5 (it was mt-3), with the same limits.
- Testnet item pass line is 0.50. Calibration with the untrained reference Qwen2.5 Coder 7B on 200 dev items passed 37.5% at 0.50, 21% at 0.60, 8% at 0.70 and 3.5% at 0.80 (mean item score 0.449). The mainnet line is set later from a frontier model and trained miners.
