# Dataset

[microtensor/agent-sessions](https://huggingface.co/datasets/microtensor/agent-sessions)

Licence: Microtensor dataset terms

Our own corpus of coding agent sessions on permissively licensed repository tasks, recorded through a tap and stripped by the canonical renderer. The dev split is public, 2,000 sessions at launch; the hidden split never leaves validators. Codex sessions join in v2.1.

Build dev items in the arena's exact format from it with

```
mta dev --arena ctx_compress --count 200 --out data/dev
```
