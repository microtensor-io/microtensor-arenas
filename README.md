# Microtensor arenas

Every arena on the network, as the files the definitions are built from. The pages here are the same ones the site shows at [microtensor.cloud/network](https://www.microtensor.cloud/network/arenas). Hidden items never live here; validators receive them only inside encrypted epoch bundles.

## Arenas

| Arena | Task | Envelope | Network |
|---|---|---|---|
| [ctx_compress-granite](ctx_compress-granite/ARENA.md) | Agent context compression on Granite 4.0 H Micro | mt-5 | testnet |
| [ctx_compress](ctx_compress/ARENA.md) | Agent context compression, archived after epoch 0 | mt-5 | testnet |
| [doc_parse](doc_parse/ARENA.md) | Documents: page parsing | mt-5 | testnet |
| [sql_gen](sql_gen/ARENA.md) | Databases: SQL generation | mt-5 | testnet |

Each folder holds:

| File | What it is |
|---|---|
| `arena.yaml` | The arena: purpose, specialised SLM, evaluation, envelope group, scoring, harness, router, escalation, floors, item counts, and the `evaluation` block (allowed hardware profiles, images by digest, probe, spot check, ban and canary numbers) |
| `schema/input.json`, `schema/output.json` | The task's input and output schemas |
| `ARENA.md` | The arena page, rendered from `arena.yaml` |
| `data/README.md` | The attached dataset and how to build dev items from it |
| `CHANGELOG.md` | What changed and when |
| `archived.yaml` | Only in an archived arena: its successor and the epochs it settled |

## Shared files

| File | What it is |
|---|---|
| `shared/envelopes.yaml` | The six envelope classes (mt-1 Budget phone, mt-2 Phone, mt-3 CPU, mt-4 AI PC, mt-5 Small GPU, mt-6 Large), which are open on testnet and mainnet, their rules and how validators measure them |
| `shared/bases.yaml` | Every allowed base model with its pinned revision, licence, size, the groups it fits and any licence condition |
| `shared/profiles.yaml` | The hardware profiles miners run their jobs on and spot checks rerun calls on, by the coordinator or by a validator with a GPU of the profile (P1 to P3, K, C): one GPU architecture, precision and batching mode each, with the calibration rule a profile must pass |

An arena names one group and may only tighten its limits. It names exactly one base model, the one miners fine tune from and are scored against; it comes from the shared list and must fit its group. An arena lists its allowed profiles from the shared list, only those the coordinator checks (P1 today), and pins every image by digest, never by tag alone. The same values are served at `GET /v2/envelopes`, `GET /v2/bases` and `GET /v2/profiles`.

Every live arena pays two pools from all of the subnet's miner emission: the specialist model pool takes 60% and the full system pool 40%, with no serving or compute share. The harness, router and escalation repair have no pools of their own and are scored inside the full system, so `item_counts` names `slm` and `system` only.

An arena id is 1 to 24 lowercase letters, digits, underscores or hyphens, starts with a letter or digit, and is the name of its folder. An arena that has settled an epoch is never redefined: a change of that size is a new arena with its own id, folder and epoch 0, and the old folder is archived with the files its last epoch ran under plus an `archived.yaml`. `mta render`, the definition builder and `mtops` skip an archived arena. ctx_compress settled epoch 0 under five pools and is archived; ctx_compress-granite is its two pool successor.

## Dev items

```
pip install -e ".[compress]"
mta dev --arena ctx_compress-granite --count 200 --out data/dev
mta check data/dev
```

`mta dev` builds items in the arena's exact format from its public dataset: the compaction instruction with its token budget, the session inside `<session>` tags, and the label the grader scores against (key facts with their aliases and probe questions with their answers). Twins are another cut of the same session. Hidden items are built the same way from fresh sessions that are never published, so train on the skill, not on these items. Pass `--items data/dev/items.jsonl` to `mt envelope check` to time your model on them.

## Rendering

`mta render` writes each `ARENA.md` and data note from its `arena.yaml`; `mta render --check` fails when one is out of date.

## Licence

Source available under the Microtensor licence in [LICENSE](LICENSE). Datasets keep their own licences.
