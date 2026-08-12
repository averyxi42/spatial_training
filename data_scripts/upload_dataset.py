#!/usr/bin/env python3
"""
Pack a tree of episode directories into uncompressed ZIP shards and upload them
to a Hugging Face dataset repo as plain blobs.

Design notes
------------
* No measuring pass. Shards close themselves when the ZIP on disk crosses the
  target size, so no per-file stat() is needed to plan boundaries.
* The episode walk is threaded and cached to disk, so a second run skips it.
* Episodes are read ahead of the ZIP writer on a thread pool, with the files
  inside an episode read concurrently. The writer stays single-threaded, so
  shard layout is byte-identical to the serial version.
* Shards are ZIP with ZIP_STORED (no compression). JPEGs don't compress, so this
  costs nothing, and the ZIP central directory means a single episode can be
  extracted later without streaming the whole shard.
* Shard boundaries fall on episode boundaries, so every shard is independently
  extractable and a failed download costs one shard, not the set.
* Several shards may upload concurrently while the next is packed. Peak extra
  disk is about (--upload-workers + 1) shards.
* Resumable via a small .idx.json sidecar uploaded next to each shard, which
  records the episodes it contains. Resume state lives in the repo, not locally.
* Ctrl-C stops after the current shard, cleans up partials, exits 130.
  A second Ctrl-C exits immediately.

Set HF_HUB_ENABLE_HF_TRANSFER=1 (after `pip install hf_transfer`) to speed up
transfer of the large shard files.

Usage
-----
  python shard_and_upload.py --scan-only          # walk and cache, no packing
  python shard_and_upload.py                      # pack and upload
  python shard_and_upload.py --unpack /dest/imgs  # fetch and restore
"""

import argparse
import concurrent.futures
import json
import os
import queue
import signal
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hf_logging
from tqdm.auto import tqdm

disable_progress_bars()
hf_logging.set_verbosity_error()

GIB = 1024 ** 3
CACHE_NAME = "episodes.txt"

CANCEL = False


def _install_sigint_handler() -> None:
    def handler(signum, frame):
        global CANCEL
        if CANCEL:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        CANCEL = True

    signal.signal(signal.SIGINT, handler)


# ----------------------------------------------------------------------------
# Episode discovery
# ----------------------------------------------------------------------------

def _scan_one(path: Path, ignore_hidden: bool) -> Tuple[Path, bool, List[Path]]:
    """
    One directory read. Returns (path, is_episode, subdirs).

    entry.is_dir() is answered from the d_type returned by the directory read
    itself on Linux and macOS, so this costs no extra syscall per file.
    Deliberately does not touch st_size.
    """
    is_episode = False
    subdirs: List[Path] = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                if ignore_hidden and entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(Path(entry.path))
                else:
                    is_episode = True
    except PermissionError:
        print(f"warning: permission denied, skipping {path}", file=sys.stderr)
    return path, is_episode, subdirs


def find_episodes(root: Path, workers: int, ignore_hidden: bool = True) -> List[Path]:
    """Breadth-first, one level at a time, directory reads in that level issued
    in parallel. Metadata operations are latency-bound, so threads help."""
    episodes: List[Path] = []
    level: List[Path] = [root]
    bar = tqdm(desc="Walking tree", unit="dir", dynamic_ncols=True, file=sys.stdout)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            while level and not CANCEL:
                next_level: List[Path] = []
                for path, is_episode, subdirs in ex.map(
                    lambda p: _scan_one(p, ignore_hidden), level
                ):
                    bar.update(1)
                    if is_episode:
                        episodes.append(path)
                    else:
                        next_level.extend(subdirs)
                level = next_level
    finally:
        bar.close()

    episodes.sort()
    return episodes


def load_or_scan(root: Path, staging: Path, workers: int, reuse: bool) -> List[Path]:
    cache = staging / CACHE_NAME
    if reuse and cache.exists():
        rels = [line for line in cache.read_text().splitlines() if line]
        print(f"Reusing cached episode list ({len(rels):,} episodes) from {cache}")
        return [root / r for r in rels]

    print(f"Walking {root} ...")
    episodes = find_episodes(root, workers)
    if not CANCEL and episodes:
        cache.write_text(
            "\n".join(ep.relative_to(root).as_posix() for ep in episodes) + "\n"
        )
        print(f"Cached episode list to {cache} (pass --reuse-scan to skip next time)")
    return episodes


# ----------------------------------------------------------------------------
# Resume state
# ----------------------------------------------------------------------------

def idx_name(shard_name: str) -> str:
    return shard_name[: -len(".zip")] + ".idx.json"


def load_completed(api: HfApi, repo_id: str, prefix: str, token) -> Tuple[Set[str], int]:
    """
    Reads the .idx.json sidecars from the repo to find which episodes are done.

    Only the longest run of sidecars starting at 0 is trusted. A shard whose
    sidecar is missing is treated as not done; re-packing reproduces it byte for
    byte (same episode order, same target) and overwrites it.
    """
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as e:
        print(f"warning: could not list repo files ({e}); starting fresh")
        return set(), 0

    done: Set[str] = set()
    index = 0
    while True:
        name = f"{prefix}-{index:04d}.idx.json"
        if name not in files:
            break
        if f"{prefix}-{index:04d}.zip" not in files:
            break
        try:
            local = hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=name, token=token
            )
            done.update(json.loads(Path(local).read_text())["episodes"])
        except Exception as e:
            print(f"warning: could not read {name} ({e}); stopping resume here")
            break
        index += 1

    if index:
        print(f"Resuming: {index} shard(s) complete, {len(done):,} episodes already stored.")
    return done, index


# ----------------------------------------------------------------------------
# Reading and packing
# ----------------------------------------------------------------------------

def _read_one(path: Path, arcname: str) -> Tuple[zipfile.ZipInfo, bytes]:
    """
    Reads one file into memory as a (ZipInfo, bytes) pair.

    ZipInfo.from_file is exactly what ZipFile.write() uses internally, so the
    local header - and therefore every subsequent offset, and therefore every
    shard boundary - is identical to what the serial version produced.
    """
    zinfo = zipfile.ZipInfo.from_file(path, arcname)
    zinfo.compress_type = zipfile.ZIP_STORED
    with open(path, "rb", buffering=0) as fh:
        return zinfo, fh.read()


class EpisodePrefetcher:
    """
    Reads episodes ahead of the writer, in order, on background threads.

    Files within an episode are read concurrently, which is where the win is on
    high-latency or networked storage. The prefetcher spans the whole run rather
    than one shard, so nothing is wasted at a boundary: whatever it has read
    next is simply the first episode of the next shard.
    """

    def __init__(self, root: Path, episodes: Sequence[Path], io_workers: int, depth: int = 2):
        self.root = root
        self.episodes = episodes
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.io = concurrent.futures.ThreadPoolExecutor(max_workers=io_workers)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _loop(self) -> None:
        try:
            for ep in self.episodes:
                if self.stop.is_set() or CANCEL:
                    break
                rel = ep.relative_to(self.root).as_posix()
                try:
                    with os.scandir(ep) as it:
                        names = sorted(
                            e.name for e in it
                            if not e.name.startswith(".")
                            and e.is_file(follow_symlinks=False)
                        )
                    items = list(self.io.map(
                        lambda n, ep=ep, rel=rel: _read_one(ep / n, f"{rel}/{n}"), names
                    ))
                except OSError as e:
                    if not self._put(("skip", rel, str(e))):
                        return
                    continue
                if not self._put(("ep", rel, items)):
                    return
        finally:
            self._put(("end", None, None))

    def get(self):
        return self.q.get()

    def close(self) -> None:
        self.stop.set()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.io.shutdown(wait=False)


def pack_until_full(
    prefetch: EpisodePrefetcher,
    out_path: Path,
    target_bytes: int,
    on_episode=None,
    note=None,
) -> Tuple[List[str], int, bool]:
    """
    Fills one shard greedily from the prefetcher.

    Returns (episode_rels_written, bytes_on_disk, exhausted). Boundary rule is
    unchanged: close once the ZIP offset passes target_bytes, always on an
    episode boundary. Writes to a .part file and renames on success.
    """
    part = out_path.with_suffix(out_path.suffix + ".part")
    if part.exists():
        part.unlink()

    written_rels: List[str] = []
    exhausted = False

    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            while True:
                if CANCEL:
                    raise KeyboardInterrupt
                kind, rel, payload = prefetch.get()
                if kind == "end":
                    exhausted = True
                    break
                if kind == "skip":
                    if note:
                        note(f"warning: skipping {rel}: {payload}")
                    continue

                for zinfo, data in payload:
                    with zf.open(zinfo, "w") as dest:
                        dest.write(data)

                written_rels.append(rel)

                # fp.tell() is the current ZIP offset: free, no stat needed.
                offset = zf.fp.tell()
                if on_episode is not None:
                    on_episode(rel, offset)
                if offset >= target_bytes:
                    break

        if not written_rels:
            part.unlink(missing_ok=True)
            return [], 0, exhausted
        part.rename(out_path)
    except BaseException:
        if part.exists():
            part.unlink()
        raise

    return written_rels, out_path.stat().st_size, exhausted


# ----------------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------------

def upload_pair(api: HfApi, repo_id: str, shard: Path, sidecar: Path, keep: bool) -> None:
    """Uploads the shard, then its sidecar. Order matters for resume."""
    api.upload_file(
        path_or_fileobj=str(shard), path_in_repo=shard.name,
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"Add {shard.name}",
    )
    api.upload_file(
        path_or_fileobj=str(sidecar), path_in_repo=sidecar.name,
        repo_id=repo_id, repo_type="dataset",
        commit_message=f"Add {sidecar.name}",
    )
    if not keep:
        shard.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------

class PackUploadUI:
    """
    Four live lines:

      Episodes    ####----  12,480/38,585 ep [00:41<02:07]
      Shard 0005  ####----  14.2G/20.0G   612 ep, 78MB/s
      Upload 0004 ###-----  est 8.6G/19.8G  3:12  46.1MB/s agg  +1 more
      Totals      4 shards | 79.1GiB sent | 0 failed

    The pack line advances every episode and shows read+write throughput. The
    upload line is an *estimate*: upload_file exposes no byte callback, so
    progress is projected from aggregate measured throughput and labelled est.
    Elapsed time is always real.

    All bar writes happen on the main thread; uploader threads never touch the
    display. Falls back to periodic one-line prints when stdout is not a tty.
    """

    def __init__(self, total_episodes: int, target_bytes: int):
        self.target = target_bytes
        self.total_eps = total_episodes
        self.tty = sys.stdout.isatty()

        self.shards_done = 0
        self.bytes_sent = 0
        self.failures = 0

        self.up_wall_start: Optional[float] = None
        self.inflight = None
        self.pack_t0 = 0.0
        self._last_line = 0.0
        self._eps = 0

        if not self.tty:
            self.overall = self.pack = self.upload = self.totals = None
            return

        common = dict(dynamic_ncols=True, file=sys.stdout, leave=True, ascii=True)
        self.overall = tqdm(
            total=total_episodes, position=0, unit="ep", unit_scale=False,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} ep "
                       "[{elapsed}<{remaining}]",
            desc="Episodes   ", **common,
        )
        self.pack = tqdm(
            total=target_bytes, position=1, unit="B", unit_scale=True, unit_divisor=1024,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}  {postfix}",
            desc="Shard  ----", **common,
        )
        self.upload = tqdm(
            total=1, position=2, unit="B", unit_scale=True, unit_divisor=1024,
            bar_format="{desc}", desc="Upload ----  idle", **common,
        )
        self.totals = tqdm(
            total=1, position=3, bar_format="{desc}",
            desc="Totals      0 shards | 0.0GiB sent | 0 failed", **common,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _clock(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def note(self, msg: str) -> None:
        if self.tty:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg, flush=True)

    def _refresh_totals(self) -> None:
        text = (f"Totals      {self.shards_done} shards | "
                f"{self.bytes_sent / GIB:.1f}GiB sent | {self.failures} failed")
        if self.totals:
            self.totals.set_description_str(text, refresh=True)

    def _fallback_line(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_line < 60:
            return
        self._last_line = now
        print(f"[{self._eps:,}/{self.total_eps:,} ep] "
              f"{self.shards_done} shards, {self.bytes_sent / GIB:.1f}GiB sent, "
              f"{self.failures} failed", flush=True)

    # -- pack phase ---------------------------------------------------------
    def start_shard(self, name: str) -> None:
        short = name.split("-")[-1].removesuffix(".zip")
        self.pack_t0 = time.monotonic()
        if self.pack:
            self.pack.reset(total=self.target)
            self.pack.set_description_str(f"Shard  {short}", refresh=False)
            self.pack.set_postfix_str("0 ep", refresh=True)

    def episode(self, in_shard: int, cum_bytes: int) -> None:
        self._eps += 1
        if not self.tty:
            self._fallback_line()
            return
        self.overall.update(1)
        self.pack.n = min(cum_bytes, self.target)
        elapsed = max(time.monotonic() - self.pack_t0, 1e-6)
        self.pack.set_postfix_str(
            f"{in_shard} ep, {cum_bytes / elapsed / 1e6:.0f}MB/s", refresh=True
        )
        self.refresh_upload()

    def shard_packed(self) -> None:
        if self.pack:
            self.pack.set_description_str("Shard  ----", refresh=False)
            self.pack.set_postfix_str("packed, waiting", refresh=True)

    # -- upload phase -------------------------------------------------------
    def attach(self, inflight) -> None:
        """Hands the UI the live deque of in-flight uploads to read from."""
        self.inflight = inflight

    def note_upload_started(self) -> None:
        if self.up_wall_start is None:
            self.up_wall_start = time.monotonic()

    def _agg_rate(self) -> Optional[float]:
        if self.up_wall_start is None or self.bytes_sent == 0:
            return None
        elapsed = time.monotonic() - self.up_wall_start
        return self.bytes_sent / elapsed if elapsed > 0 else None

    def refresh_upload(self) -> None:
        if not self.upload:
            return
        inflight = self.inflight
        if not inflight:
            self.upload.bar_format = "{desc}"
            self.upload.set_description_str("Upload ----  idle", refresh=True)
            return

        _, name, size, t0 = inflight[0]
        short = name.split("-")[-1].removesuffix(".zip")
        elapsed = time.monotonic() - t0
        extra = f"  +{len(inflight) - 1} more" if len(inflight) > 1 else ""
        rate = self._agg_rate()

        if rate:
            # Aggregate throughput, shared across whatever is in flight.
            share = rate / len(inflight)
            self.upload.total = size
            self.upload.n = int(min(share * elapsed, size * 0.99))
            self.upload.bar_format = (
                "{desc} {percentage:3.0f}%|{bar}| est {n_fmt}/{total_fmt}  {postfix}"
            )
            self.upload.set_postfix_str(
                f"{self._clock(elapsed)}  {rate / 1e6:.1f}MB/s agg{extra}", refresh=False
            )
            self.upload.set_description_str(f"Upload {short}", refresh=True)
        else:
            self.upload.bar_format = "{desc}"
            self.upload.set_description_str(
                f"Upload {short}  {size / GIB:.1f}GiB  {self._clock(elapsed)}"
                f"  (no rate estimate yet){extra}",
                refresh=True,
            )

    def finish_upload(self, ok: bool, size: int) -> None:
        if ok:
            self.shards_done += 1
            self.bytes_sent += size
        else:
            self.failures += 1
        self._refresh_totals()
        self.refresh_upload()
        if not self.tty:
            self._fallback_line(force=True)

    def close(self) -> None:
        for bar in (self.overall, self.pack, self.upload, self.totals):
            if bar:
                bar.close()
        if self.tty:
            print()


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    staging = Path(args.staging).resolve()
    staging.mkdir(parents=True, exist_ok=True)

    episodes = load_or_scan(root, staging, args.scan_workers, args.reuse_scan)
    if CANCEL:
        print("\nInterrupted during the walk; nothing written.")
        return 130
    if not episodes:
        print("No episode directories found.")
        return 0
    print(f"{len(episodes):,} episode directory/ies.")

    if args.scan_only:
        print("Scan only; stopping before packing.")
        return 0

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset",
        exist_ok=True, private=args.private,
    )

    done, shard_index = load_completed(api, args.repo_id, args.prefix, args.token)
    remaining = [ep for ep in episodes if ep.relative_to(root).as_posix() not in done]
    if not remaining:
        print("Everything is already uploaded.")
        return 0

    failures: List[Tuple[str, str]] = []
    uploaded = 0
    total_bytes = 0
    interrupted = False

    ui = PackUploadUI(len(remaining), args.shard_bytes)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.upload_workers)
    inflight = deque()
    ui.attach(inflight)
    prefetch = EpisodePrefetcher(root, remaining, args.read_workers)

    def collect() -> None:
        """Retires the oldest in-flight upload, keeping the display ticking."""
        nonlocal uploaded
        future, name, size, _ = inflight[0]
        while True:
            try:
                future.result(timeout=0.5)
                inflight.popleft()
                uploaded += 1
                ui.finish_upload(True, size)
                return
            except concurrent.futures.TimeoutError:
                ui.refresh_upload()
            except Exception as e:
                inflight.popleft()
                failures.append((name, str(e)))
                ui.finish_upload(False, size)
                ui.note(f"[FAILED] {name} | {e}")
                return

    try:
        while True:
            if CANCEL:
                interrupted = True
                break

            name = f"{args.prefix}-{shard_index:04d}.zip"
            shard_path = staging / name
            ui.start_shard(name)
            packed_here = 0

            def on_episode(rel: str, cum_bytes: int) -> None:
                nonlocal packed_here
                packed_here += 1
                ui.episode(packed_here, cum_bytes)

            try:
                rels, size, exhausted = pack_until_full(
                    prefetch, shard_path, args.shard_bytes,
                    on_episode=on_episode, note=ui.note,
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except OSError as e:
                failures.append((name, f"pack failed: {e}"))
                ui.note(f"[FAILED] {name} | pack failed: {e}")
                break

            if rels:
                sidecar = staging / idx_name(name)
                sidecar.write_text(json.dumps({"shard": name, "episodes": rels}))
                total_bytes += size
                ui.shard_packed()

                # Cap concurrent uploads; peak disk stays near
                # (upload_workers + 1) shards.
                while len(inflight) >= args.upload_workers:
                    collect()

                inflight.append((
                    executor.submit(upload_pair, api, args.repo_id, shard_path,
                                    sidecar, args.keep_shards),
                    name, size, time.monotonic(),
                ))
                ui.note_upload_started()
                ui.refresh_upload()
                shard_index += 1

            if exhausted:
                break

        if inflight:
            if interrupted:
                ui.note("Interrupted; finishing in-flight upload(s) "
                        "(Ctrl-C again to abandon them)...")
            while inflight:
                collect()
    except KeyboardInterrupt:
        interrupted = True
        ui.note("Forced exit; abandoning in-flight upload(s).")
    finally:
        prefetch.close()
        ui.close()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    print(f"\nUploaded {uploaded} shard(s), {total_bytes / GIB:.1f} GiB this run.")
    if failures:
        print(f"{len(failures)} failure(s):")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
    if interrupted:
        print("Interrupted. Re-run with --reuse-scan to resume.")
        return 130
    if failures:
        print("Re-run with --reuse-scan to retry.")
        return 1
    print("Done.")
    return 0


# ----------------------------------------------------------------------------
# Unpack
# ----------------------------------------------------------------------------

def unpack(args: argparse.Namespace) -> int:
    dest = Path(args.unpack).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cache = Path(args.staging).resolve() / "download"

    print(f"Downloading {args.repo_id} -> {cache}")
    local = Path(snapshot_download(
        repo_id=args.repo_id, repo_type="dataset",
        local_dir=str(cache), token=args.token,
    ))

    shards = sorted(local.glob("*.zip"))
    if not shards:
        print("No .zip shards found in the repo.", file=sys.stderr)
        return 1

    missing = [s for s in shards if not (s.parent / idx_name(s.name)).exists()]
    if missing:
        print(f"warning: {len(missing)} shard(s) have no sidecar; upload may be incomplete")

    bar = tqdm(
        total=sum(p.stat().st_size for p in shards), desc="Extracting",
        unit="B", unit_scale=True, unit_divisor=1024,
        dynamic_ncols=True, file=sys.stdout, ascii=True,
    )
    failures: List[Tuple[str, str]] = []
    try:
        for shard in shards:
            if CANCEL:
                bar.write("Interrupted; stopping between shards.")
                break
            try:
                with zipfile.ZipFile(shard) as zf:
                    zf.extractall(dest)
            except Exception as e:
                failures.append((shard.name, str(e)))
                bar.write(f"[FAILED] {shard.name} | {e}")
            bar.update(shard.stat().st_size)
    finally:
        bar.close()

    if failures:
        print(f"{len(failures)} shard(s) failed to extract:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1

    print(f"Extracted to {dest}")
    print(f"Downloaded shards remain in {cache}; remove when satisfied.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="/Projects/habitat_physical_nav/recordings/images")
    p.add_argument("--repo-id", default="Aasdfip/continuous-habitat-web-demo-images")
    p.add_argument("--prefix", default="images", help="Shard filename prefix.")
    p.add_argument("--staging", default="./_shards",
                   help="Scratch dir for shards in flight.")
    p.add_argument("--shard-gib", type=float, default=20.0,
                   help="Target shard size in GiB. Must match earlier runs to resume.")
    p.add_argument("--read-workers", type=int, default=8,
                   help="Threads reading JPEGs ahead of the ZIP writer.")
    p.add_argument("--upload-workers", type=int, default=2,
                   help="Shards uploading concurrently. Peak disk ~ this + 1 shards.")
    p.add_argument("--scan-workers", type=int, default=32,
                   help="Threads for the directory walk. Raise on networked storage.")
    p.add_argument("--reuse-scan", action="store_true",
                   help="Reuse the cached episode list instead of re-walking.")
    p.add_argument("--scan-only", action="store_true",
                   help="Walk and cache the episode list, then stop.")
    p.add_argument("--keep-shards", action="store_true",
                   help="Keep local shards after upload instead of deleting each one.")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    p.add_argument("--token", default=None, help="HF token; defaults to cached login.")
    p.add_argument("--unpack", metavar="DEST",
                   help="Download the repo and extract all shards into DEST.")
    args = p.parse_args()
    args.shard_bytes = int(args.shard_gib * GIB)

    _install_sigint_handler()

    if args.unpack:
        return unpack(args)
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)