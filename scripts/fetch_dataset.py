"""Restore the parsed dataset from the private HF repo onto any machine.

Downloads shard tars one at a time, extracts into Root/Parsed, deletes each
tar immediately (peak disk = dataset size + one shard). Resumable: already-
extracted shards are detected via a marker file per shard.

Requires a read token for the private repo (env HF_TOKEN or HF cli login).

Usage (on a cloud box):
  python scripts/fetch_dataset.py --root /workspace/data/full/Root \
      [--repo KeithG33/shinebot-melee-parsed]
"""

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo", default="KeithG33/shinebot-melee-parsed")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    all_files = api.list_repo_files(args.repo, repo_type="dataset")
    shard_names = sorted(f for f in all_files if f.startswith("shards/"))
    assert "meta.json" in all_files, "meta.json missing: upload incomplete?"

    parsed_dir = args.root / "Parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    markers = args.root / ".fetched"
    markers.mkdir(exist_ok=True)

    for name in shard_names:
        marker = markers / Path(name).name
        if marker.exists():
            print(f"skip {name} (already extracted)")
            continue
        tar_path = Path(
            hf_hub_download(args.repo, name, repo_type="dataset",
                            local_dir=str(args.root / ".dl"))
        )
        with tarfile.open(tar_path) as tar:
            tar.extractall(parsed_dir, filter="data")
        tar_path.unlink()
        marker.touch()
        print(f"extracted {name}", flush=True)

    for fname in ["parsed.sqlite", "parsed.pkl", "raw.json", "meta.json"]:
        if fname in all_files:
            path = hf_hub_download(args.repo, fname, repo_type="dataset",
                                   local_dir=str(args.root / ".dl"))
            (args.root / fname).write_bytes(Path(path).read_bytes())

    n = sum(1 for _ in parsed_dir.iterdir())
    print(f"DONE: {n} games in {parsed_dir}")


if __name__ == "__main__":
    main()
