# Microtensor arenas

Every arena on the network, as the files the definitions are built from. The pages here are the same ones the site shows at [microtensor.cloud/network](https://www.microtensor.cloud/network/arenas). Hidden items never live here; validators receive them only inside encrypted epoch bundles.

## Arenas

| Arena | Task | Envelope | Network |
|---|---|---|---|
| [ctx_compress](ctx_compress/ARENA.md) | Agent context compression | mt-5 | testnet |
| [sql_gen](sql_gen/ARENA.md) | Databases: SQL generation | mt-5 | testnet |

Each folder holds:

| File | What it is |
|---|---|
| `arena.yaml` | The arena: purpose, specialised SLM, evaluation, envelope group, scoring, harness, router, escalation, floors, item counts |
| `schema/input.json`, `schema/output.json` | The task's input and output schemas |
| `ARENA.md` | The arena page, rendered from `arena.yaml` |
| `data/README.md` | The attached dataset and how to build dev items from it |
| `CHANGELOG.md` | What changed and when |

## Shared files

| File | What it is |
|---|---|
| `shared/envelopes.yaml` | The six envelope classes (mt-1 Budget phone, mt-2 Phone, mt-3 CPU, mt-4 AI PC, mt-5 Small GPU, mt-6 Large), which are open on testnet and mainnet, their rules and how validators measure them |
| `shared/bases.yaml` | Every allowed base model with its pinned revision, licence, size, the groups it fits and any licence condition |

An arena names one group and may only tighten its limits. Its bases come from the shared list and must fit its group. The same values are served at `GET /v2/envelopes` and `GET /v2/bases`.

## Dev items

```
pip install -e ".[compress]"
mta dev --arena ctx_compress --count 200 --out data/dev
mta check data/dev
```

`mta dev` builds items in the arena's exact format from its public dataset: the compaction instruction with its token budget, the session inside `<session>` tags, and the label the grader scores against (key facts with their aliases and probe questions with their answers). Twins are another cut of the same session. Hidden items are built the same way from fresh sessions that are never published, so train on the skill, not on these items. Pass `--items data/dev/items.jsonl` to `mt envelope check` to time your model on them.

## Rendering

`mta render` writes each `ARENA.md` and data note from its `arena.yaml`; `mta render --check` fails when one is out of date.

## Licence

Source available under the Microtensor licence in [LICENSE](LICENSE). Datasets keep their own licences.
