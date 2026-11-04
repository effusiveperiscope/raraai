#!/usr/bin/env python3
"""Shard a directory tree into tar archives (optionally gz/zstd compressed).

Examples:
  # ~2 GiB shards, shuffled, 8 parallel workers
  shard.py /data /out --target-size 2G --shuffle --workers 8

  # fixed 2000 files per shard, zstd-compressed
  shard.py /data /out --max-files 2000 --compress zstd --level 3

  # preview the plan without writing anything
  shard.py /data /out --target-size 2G --dry-run
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

EXT = {"none": ".tar", "gz": ".tar.gz", "zstd": ".tar.zst"}


def parse_size(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?", s.strip(), re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r} (try 500M, 2G)")
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(m.group(1)) * mult[m.group(2).upper()])


def fmt_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def scan(src, follow_symlinks):
    files = []
    for root, _dirs, names in os.walk(src, followlinks=follow_symlinks):
        for name in names:
            full = os.path.join(root, name)
            try:
                st = os.stat(full) if follow_symlinks else os.lstat(full)
            except OSError as e:
                print(f"skip (stat failed): {full}: {e}", file=sys.stderr)
                continue
            if not os.path.isfile(full) or (not follow_symlinks and os.path.islink(full)):
                continue
            files.append((os.path.relpath(full, src), st.st_size))
    return files


def group(files, max_files, target_size):
    shards, cur, cur_size = [], [], 0
    for rel, size in files:
        cur.append((rel, size))
        cur_size += size
        if (max_files and len(cur) >= max_files) or (target_size and cur_size >= target_size):
            shards.append(cur)
            cur, cur_size = [], 0
    if cur:
        shards.append(cur)
    return shards


def write_shard(job):
    idx, files, src, out_path, compress, level, verify = job
    t0 = time.time()
    tmp = out_path + ".part"
    try:
        if compress == "zstd":
            proc = subprocess.Popen(
                ["zstd", "-T1", f"-{level}", "-q", "-f", "-o", tmp],
                stdin=subprocess.PIPE,
            )
            tf = tarfile.open(fileobj=proc.stdin, mode="w|")
        elif compress == "gz":
            tf = tarfile.open(tmp, mode="w:gz", compresslevel=level)
            proc = None
        else:
            tf = tarfile.open(tmp, mode="w")
            proc = None

        with tf:
            for rel, _size in files:
                tf.add(os.path.join(src, rel), arcname=rel, recursive=False)

        if proc is not None:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("zstd failed")

        if verify:
            n = count_members(tmp, compress)
            if n != len(files):
                raise RuntimeError(f"verify failed: {n} members, expected {len(files)}")

        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return idx, out_path, len(files), os.path.getsize(out_path), time.time() - t0


def count_members(path, compress):
    if compress == "zstd":
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            n = sum(1 for _ in tf)
        proc.stdout.close()
        proc.wait()
        return n
    with tarfile.open(path, mode="r:*") as tf:
        return sum(1 for _ in tf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="source directory")
    p.add_argument("out", help="output directory for shards")
    p.add_argument("--max-files", type=int, default=0, help="max files per shard")
    p.add_argument("--target-size", type=parse_size, default=0,
                   help="close a shard once it reaches this much input data (e.g. 2G)")
    p.add_argument("--shuffle", action="store_true", help="shuffle file order before sharding")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    p.add_argument("--compress", choices=["none", "gz", "zstd"], default="none")
    p.add_argument("--level", type=int, default=None, help="compression level (gz default 6, zstd default 3)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="parallel tar jobs")
    p.add_argument("--prefix", default="shard", help="shard filename prefix")
    p.add_argument("--manifest", default="manifest.tsv", help="manifest filename (written inside out dir)")
    p.add_argument("--follow-symlinks", action="store_true", help="follow symlinks instead of skipping them")
    p.add_argument("--skip-existing", action="store_true", help="skip shards whose output file already exists (resume)")
    p.add_argument("--verify", action="store_true", help="re-read each shard and check member count")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = p.parse_args()

    if not os.path.isdir(a.src):
        p.error(f"not a directory: {a.src}")
    if not a.max_files and not a.target_size:
        a.max_files = 2000
    if a.level is None:
        a.level = 6 if a.compress == "gz" else 3
    src = os.path.abspath(a.src)

    print(f"scanning {src} ...", flush=True)
    files = scan(src, a.follow_symlinks)
    if not files:
        sys.exit("no files found")
    if a.shuffle:
        random.Random(a.seed).shuffle(files)
    else:
        files.sort()

    shards = group(files, a.max_files, a.target_size)
    total = sum(s for _, s in files)
    width = max(4, len(str(len(shards) - 1)))
    ext = EXT[a.compress]
    names = [f"{a.prefix}-{i:0{width}d}{ext}" for i in range(len(shards))]

    print(f"{len(files)} files, {fmt_size(total)} -> {len(shards)} shards "
          f"(~{fmt_size(total // len(shards))} / {len(files) // len(shards)} files each)")
    if a.dry_run:
        return

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.manifest), "w") as m:
        m.write("shard\tpath\tsize\n")
        for name, shard in zip(names, shards):
            for rel, size in shard:
                m.write(f"{name}\t{rel}\t{size}\n")

    jobs = []
    for i, (name, shard) in enumerate(zip(names, shards)):
        out_path = os.path.join(a.out, name)
        if a.skip_existing and os.path.exists(out_path):
            continue
        jobs.append((i, shard, src, out_path, a.compress, a.level, a.verify))

    done, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(write_shard, j) for j in jobs]
        for f in as_completed(futs):
            idx, path, n, size, secs = f.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {os.path.basename(path)}: {n} files, "
                  f"{fmt_size(size)} in {secs:.1f}s", flush=True)

    print(f"done in {time.time() - t0:.0f}s. manifest: {os.path.join(a.out, a.manifest)}")


if __name__ == "__main__":
    main()