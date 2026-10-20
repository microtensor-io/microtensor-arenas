# Dataset

[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)

Licence: CC BY 4.0

Dev split only; hidden items never go in the repo. 80,036 SWE-agent trajectories written by Llama 3.1 70B. The card says use of those outputs follows the Llama 3.1 licence, so a distributed model trained on them must start its name with "Llama" and show "Built with Llama".

Build dev items in the arena's exact format from it with

```
mta dev --arena ctx_compress --count 200 --out data/dev
```
