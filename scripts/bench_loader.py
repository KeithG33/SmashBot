"""Benchmark the torch data bridge on a parsed dataset.

Must be a real file (not stdin): DataSource's forkserver workers re-import
__main__. Usage: python scripts/bench_loader.py [--root DIR] [--num_workers N]
"""

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kage/drive2/ShineBot/data/full/Root")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--unroll_length", type=int, default=80)
    ap.add_argument("--extra_frames", type=int, default=19)  # delay 18 + 1
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--num_batches", type=int, default=20)
    args = ap.parse_args()

    from slippi_ai.data import DatasetConfig
    from shinebot.configs import DataConfig
    from shinebot.data import loader

    cfg = DataConfig(
        dataset=DatasetConfig(
            data_dir=f"{args.root}/Parsed",
            meta_path=f"{args.root}/meta-20k.json",
            allowed_characters="fox",
            allowed_opponents="all",
        ),
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_workers=args.num_workers,
        prefetch=4,
    )
    sources = loader.make_sources(cfg, extra_frames=args.extra_frames)
    print("name_map:", sources.name_map)
    stream = loader.TorchBatchStream(sources.train, cfg)
    try:
        for _ in range(3):  # warmup
            next(stream)
        t0 = time.perf_counter()
        for _ in range(args.num_batches):
            batch, epoch = next(stream)
        dt = time.perf_counter() - t0
        chunk = args.unroll_length + args.extra_frames
        frames = args.num_batches * args.batch_size * chunk
        print(
            f"{args.num_batches / dt:.2f} batches/s | {frames / dt / 1e6:.2f}M frames/s"
            f" | epoch={epoch:.3f}"
        )
        print(
            "stage shape:", tuple(batch.game.stage.shape),
            "| pinned:", batch.game.stage.is_pinned(),
        )
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
