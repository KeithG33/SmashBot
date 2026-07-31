"""Build the M1 debug corpus: take N .slp files from a downloaded HF ranked
shard (.tar.gz), pack them into a zip under <root>/Raw/, then hand off to
slippi_db/parse_local.py and make_local_dataset.py (run separately).

Usage (training venv):
  python scripts/make_debug_corpus.py --shard <path.tar.gz> --root <Root dir> [--num 300]
"""

import argparse
import tarfile
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--num", type=int, default=300)
    args = ap.parse_args()

    raw_dir = args.root / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_zip = raw_dir / f"{args.shard.stem.replace('.tar', '')}_first{args.num}.zip"

    count = 0
    with tarfile.open(args.shard, "r:gz") as tar, zipfile.ZipFile(
        out_zip, "w", zipfile.ZIP_STORED
    ) as zf:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".slp"):
                continue
            fobj = tar.extractfile(member)
            assert fobj is not None
            zf.writestr(Path(member.name).name, fobj.read())
            count += 1
            if count >= args.num:
                break

    print(f"wrote {count} .slp files to {out_zip}")


if __name__ == "__main__":
    main()
