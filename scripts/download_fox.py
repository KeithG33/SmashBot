"""M7 dataset builder: pipelined download -> repack -> parse for all Fox shards.

Producer threads download tar.gz shards from HuggingFace and repack the .slp
files into stored zips under <root>/Raw/. The main loop runs slippi_db's
parse_local over completed zips (all CPU threads) and deletes raw archives as
they finish, so peak disk stays at ~a few shards + the (small) parsed output.

Resumable: already-parsed shards are recorded in <root>/raw.json and skipped;
interrupted downloads resume via huggingface_hub.

Usage:
  .venv/bin/python scripts/download_fox.py                  # all 36 Fox shards
  .venv/bin/python scripts/download_fox.py --limit 2        # trial run
"""

import argparse
import json
import queue
import subprocess
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

REPO = "erickfm/melee-ranked-replays"
VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "slippi-ai"


def list_fox_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("FOX/") and f.endswith(".tar.gz"))


def shard_zip_name(shard: str) -> str:
    return Path(shard).name.replace(".tar.gz", ".zip")


def already_processed(root: Path) -> set[str]:
    raw_json = root / "raw.json"
    if not raw_json.exists():
        return set()
    meta = json.loads(raw_json.read_text())
    return {row["name"] for row in meta if row.get("processed")}


def download_and_repack(shard: str, root: Path, dl_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    tar_path = Path(
        hf_hub_download(REPO, shard, repo_type="dataset", local_dir=str(dl_dir))
    )
    zip_path = root / "Raw" / shard_zip_name(shard)
    tmp_path = zip_path.with_suffix(".zip.tmp")

    count = 0
    with tarfile.open(tar_path, "r:gz") as tar, zipfile.ZipFile(
        tmp_path, "w", zipfile.ZIP_STORED
    ) as zf:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".slp"):
                continue
            fobj = tar.extractfile(member)
            assert fobj is not None
            zf.writestr(Path(member.name).name, fobj.read())
            count += 1

    tmp_path.rename(zip_path)
    tar_path.unlink()  # compressed original no longer needed
    return f"{zip_path.name}: {count} games"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/home/kage/drive2/ShineBot/data/fox-full/Root"))
    ap.add_argument("--dl_dir", type=Path,
                    default=Path("/home/kage/drive2/ShineBot/data/hf-raw"))
    ap.add_argument("--downloaders", type=int, default=2)
    ap.add_argument("--parse_threads", type=int, default=40)
    ap.add_argument("--max_pending_zips", type=int, default=3,
                    help="disk guard: downloader stalls if this many unparsed zips")
    ap.add_argument("--limit", type=int, default=0, help="only first N shards (testing)")
    ap.add_argument("--keep_raw", action="store_true")
    args = ap.parse_args()

    (args.root / "Raw").mkdir(parents=True, exist_ok=True)

    shards = list_fox_shards()
    if args.limit:
        shards = shards[: args.limit]
    done = already_processed(args.root)
    todo = [s for s in shards if shard_zip_name(s) not in done]
    print(f"{len(shards)} shards total; {len(shards) - len(todo)} already parsed; "
          f"{len(todo)} to go")

    work: queue.Queue = queue.Queue()
    for s in todo:
        work.put(s)
    ready = threading.Semaphore(args.max_pending_zips)
    errors: list[str] = []

    def producer():
        while True:
            try:
                shard = work.get_nowait()
            except queue.Empty:
                return
            ready.acquire()  # wait if too many unparsed zips on disk
            try:
                msg = download_and_repack(shard, args.root, args.dl_dir)
                print(f"[dl] {msg}", flush=True)
            except Exception as e:
                errors.append(f"{shard}: {e}")
                print(f"[dl] FAILED {shard}: {e}", file=sys.stderr, flush=True)
                ready.release()

    producers = [threading.Thread(target=producer, daemon=True)
                 for _ in range(args.downloaders)]
    for p in producers:
        p.start()

    python = sys.executable

    def parse_pending() -> int:
        """Run parse_local over whatever zips are currently in Raw."""
        pending = list((args.root / "Raw").glob("*.zip"))
        if not pending:
            return 0
        subprocess.run(
            [python, str(VENDOR / "slippi_db" / "parse_local.py"),
             f"--root={args.root}", f"--threads={args.parse_threads}"],
            check=True,
        )
        processed = already_processed(args.root)
        n = 0
        for z in pending:
            if z.name in processed:
                if not args.keep_raw:
                    z.unlink()
                ready.release()  # free a download slot
                n += 1
        return n

    parsed_count = len(shards) - len(todo)
    while any(p.is_alive() for p in producers) or list((args.root / "Raw").glob("*.zip")):
        n = parse_pending()
        parsed_count += n
        if n:
            print(f"[parse] {parsed_count}/{len(shards)} shards done", flush=True)
        else:
            threading.Event().wait(10)  # nothing ready yet; poll

    print("finalizing dataset metadata...")
    subprocess.run(
        [python, "-m", "slippi_db.scripts.convert_sqlite_to_parsed",
         f"--root={args.root}"],
        check=True, cwd=str(VENDOR),
    )
    subprocess.run(
        [python, str(VENDOR / "slippi_db" / "scripts" / "make_local_dataset.py"),
         f"--root={args.root}"],
        check=True,
    )

    meta = json.loads((args.root / "meta.json").read_text())
    print(f"\nDONE: {len(meta)} training games in {args.root}/meta.json")
    if errors:
        print(f"{len(errors)} shard failures (rerun to retry):")
        for e in errors:
            print("  ", e)


if __name__ == "__main__":
    main()
