from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS = {
    "openhands": {
        "repo": "nebius/SWE-rebench-openhands-trajectories",
        "prefix": "data/",
        "columns": ["instance_id", "trajectory", "resolved", "repo", "created_at", "exit_status"],
        "licence": "CC BY 4.0",
    },
    "coderforge": {
        "repo": "togethercomputer/CoderForge-Preview",
        "prefix": "trajectories/SWE_Rebench",
        "columns": ["trajectory_id", "messages", "reward", "image", "license", "finish_reason"],
        "licence": "per row (repository licence); keep permissive rows only",
    },
}
PERMISSIVE = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
    "mpl-2.0",
}


def _keep(source: str, row: dict[str, Any]) -> bool:
    if source == "coderforge":
        return (
            str(row.get("license") or "").strip().lower() in PERMISSIVE
            and str(row.get("finish_reason") or "") != "error"
        )
    return True


def fetch(source: str, out: Path, limit: int | None = None, shards: int | None = None) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    spec = DATASETS[source]
    names = sorted(
        n
        for n in HfApi().list_repo_files(spec["repo"], repo_type="dataset")
        if n.startswith(spec["prefix"]) and n.endswith(".parquet")
    )
    if shards:
        names = names[:shards]
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in names:
        path = Path(hf_hub_download(spec["repo"], name, repo_type="dataset"))
        file = pq.ParquetFile(path)
        columns = [c for c in spec["columns"] if c in file.schema.names]
        target = out / (Path(name).stem + ".jsonl")
        with target.open("w", encoding="utf-8") as handle:
            for batch in file.iter_batches(batch_size=32, columns=columns):
                for row in batch.to_pylist():
                    if not _keep(source, row):
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if limit and written >= limit:
                        return written
    return written
