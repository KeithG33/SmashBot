"""Upload the parsed dataset to a public HF dataset repo as tar shards.

HF repos degrade past ~100k files, and Parsed/ holds 1.578M parquets — so
games are packed into ~32 uncompressed tars (~4GB each, parquet is already
zlib'd) and uploaded one at a time. Resumable: shards already in the repo are
skipped. meta.json is uploaded LAST and acts as the completion marker.

Requires a WRITE token (env HF_TOKEN or $HF_HOME/token).

Usage:
  .venv/bin/python scripts/upload_dataset.py [--repo KeithG33/shinebot-melee-parsed]
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

GAMES_PER_SHARD = 50_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/home/kage/drive2/ShineBot/data/full/Root"))
    ap.add_argument("--repo", default="KeithG33/shinebot-melee-parsed")
    ap.add_argument("--staging", type=Path,
                    default=Path("/home/kage/drive2/ShineBot/data/upload-staging"))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))

    files = sorted(p.name for p in (args.root / "Parsed").iterdir())
    shards = [files[i : i + GAMES_PER_SHARD] for i in range(0, len(files), GAMES_PER_SHARD)]
    print(f"{len(files)} games -> {len(shards)} shards; "
          f"{len(existing)} files already in repo")

    args.staging.mkdir(parents=True, exist_ok=True)
    for i, shard_files in enumerate(shards):
        name = f"shards/shard_{i:03d}.tar"
        if name in existing:
            print(f"skip {name} (already uploaded)")
            continue
        tar_path = args.staging / f"shard_{i:03d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for fname in shard_files:
                tar.add(args.root / "Parsed" / fname, arcname=fname)
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        tar_path.unlink()
        print(f"uploaded {name} ({len(shard_files)} games)", flush=True)

    # metadata; meta.json last = completion marker
    for fname in ["parsed.sqlite", "parsed.pkl", "raw.json", "meta.json"]:
        path = args.root / fname
        if path.exists() and fname not in existing:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=fname,
                repo_id=args.repo,
                repo_type="dataset",
            )
            print(f"uploaded {fname}", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
