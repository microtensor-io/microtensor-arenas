# Changelog

- Moved into this folder from code. Qwen2.5 Coder 3B is no longer listed as a base: its licence is not Apache 2.0 or MIT, so it could never pass the licence check.
- Envelope classes version 2: the Small GPU class is now mt-5 (it was mt-3), with the same limits.
- Renamed from sql_gen_s1. An arena runs on testnet or mainnet.
- Miner mode evaluation: miners run their own evaluation on hardware profile P1 (RTX 4090) inside the pinned images and upload a transcript; validators replay it on CPU, spot check at least 200 SLM calls per sampled miner and grade by SQLite execution on CPU (pure CPU grader; the probe entry is carried for the shared block). The settle margin rises to 1,800 blocks (6 h) so grading fits. The runner, replayer and pinned llama.cpp image digests are placeholders until those images are built.
