#!/usr/bin/env python3
"""Reshard a single-file safetensors snapshot into N-byte shards.

Byte-preserving on tensor data; writes model-XXXXX-of-YYYYY.safetensors plus
model.safetensors.index.json; copies every other snapshot file unchanged and
omits the original model.safetensors. With --scramble, tensors are assigned
to shards round-robin instead of contiguously, so shard-local order differs
from the original file order.

Usage: reshard.py <src_dir> <dst_dir> --max-shard-bytes N [--scramble]
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-shard-bytes", type=int, default=200 * 1024 * 1024)
    ap.add_argument("--scramble", action="store_true")
    args = ap.parse_args()

    args.dst.mkdir(parents=True)
    for f in args.src.iterdir():
        if f.name == "model.safetensors" or f.name.startswith("."):
            continue
        if f.is_file():
            shutil.copy2(f, args.dst / f.name)

    tensors = load_file(args.src / "model.safetensors")
    names = list(tensors.keys())

    nbytes = {n: tensors[n].numel() * tensors[n].element_size() for n in names}
    if args.scramble:
        total = sum(nbytes.values())
        n_shards = max(2, -(-total // args.max_shard_bytes))
        shards = [[] for _ in range(n_shards)]
        for i, n in enumerate(names):
            shards[i % n_shards].append(n)
    else:
        shards, current, size = [], [], 0
        for n in names:
            if current and size + nbytes[n] > args.max_shard_bytes:
                shards.append(current)
                current, size = [], 0
            current.append(n)
            size += nbytes[n]
        if current:
            shards.append(current)

    weight_map = {}
    for i, shard_names in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({n: tensors[n] for n in shard_names}, args.dst / fname,
                  metadata={"format": "pt"})
        for n in shard_names:
            weight_map[n] = fname

    index = {
        "metadata": {"total_size": sum(nbytes.values())},
        "weight_map": weight_map,
    }
    (args.dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"{len(shards)} shards, {len(names)} tensors, scramble={args.scramble}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
