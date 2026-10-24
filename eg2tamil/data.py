"""Prepared shards -> batches.

A shard is a parquet file with columns id, audio (FLAC 16 kHz mono PCM_16), text, duration.
The source name is the name of the parent directory of the shard.
"""

import glob
import io
import os
import random
import zlib

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .units import Units

FRAME_RATE = 25  # encoder output frames per second
_COLS = ["audio", "text", "duration"]


def list_shards(data_dir: str, split: str, sources: list[str] | None = None) -> dict[str, list[str]]:
    """Return {source: sorted parquet paths} for data_dir/<split>/<source>/*.parquet."""
    out = {}
    for d in sorted(glob.glob(os.path.join(data_dir, split, "*"))):
        name = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if files and (not sources or name in sources):
            out[name] = files
    return out


def _read(path: str, row_group: int | None = None):
    """Return a list of (flac_bytes, text, duration) rows."""
    f = pq.ParquetFile(path)
    t = f.read(columns=_COLS) if row_group is None else f.read_row_group(row_group, columns=_COLS)
    return list(zip(*(t.column(c).to_pylist() for c in _COLS)))


def _collate(rows: list, targets: list[list[int]] | None = None) -> dict:
    """Decode the FLAC bytes of `rows` and pad them into one batch."""
    wavs = [sf.read(io.BytesIO(r[0]), dtype="int16")[0] for r in rows]
    lens = torch.tensor([len(w) for w in wavs])
    wav = torch.zeros(len(wavs), int(lens.max()))
    for i, w in enumerate(wavs):
        wav[i, : len(w)] = torch.from_numpy(w.astype(np.float32) / 32768.0)
    batch = {"wav": wav, "wav_lens": lens, "texts": [r[1] for r in rows]}
    if targets is not None:
        batch["target_lens"] = torch.tensor([len(t) for t in targets])
        batch["targets"] = torch.tensor([u for t in targets for u in t])
    return batch


class TrainStream(IterableDataset):
    """Infinite stream of dynamic batches. Create the DataLoader with batch_size=None."""

    def __init__(self, files, units_path, seed=0, max_batch_seconds=600.0, max_batch_size=512,
                 buffer_utts=4000, source_weights: dict[str, float] | None = None):
        self.units_path, self.seed = units_path, seed
        self.max_batch_seconds, self.max_batch_size, self.buffer_utts = max_batch_seconds, max_batch_size, buffer_utts
        # Work items are (file, row_group). A source weight w repeats each item int(w) times,
        # plus one more time with chance w - int(w). The chance is a fixed hash, so it is the same in every worker.
        self.items = []
        for f in files:
            w = (source_weights or {}).get(os.path.basename(os.path.dirname(f)), 1.0)
            for rg in range(pq.ParquetFile(f).metadata.num_row_groups):
                extra = random.Random(zlib.crc32(f"{f}:{rg}:{seed}".encode())).random() < w % 1
                self.items += [(f, rg)] * (int(w) + extra)

    def _batches(self, buf: list, rng: random.Random):
        """Sort rows by duration, cut them into batches of limited padded seconds, shuffle the batches."""
        buf.sort(key=lambda r: r[0][2])
        batches, cur, longest = [], [], 0.0
        for r in buf:
            if cur and (max(longest, r[0][2]) * (len(cur) + 1) > self.max_batch_seconds
                        or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, longest = [], 0.0
            cur.append(r)
            longest = max(longest, r[0][2])
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        for b in batches:
            yield _collate([r[0] for r in b], [r[1] for r in b])

    def __iter__(self):
        units = Units.load(self.units_path)
        wi = get_worker_info()
        n, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        epoch = 0
        while True:
            order = list(range(len(self.items)))
            random.Random(self.seed + epoch).shuffle(order)  # same order in every worker
            rng = random.Random(self.seed * 1000003 + epoch * 7919 + wid)
            buf = []
            if not order[wid::n]:
                return
            for i in order[wid::n]:
                rows = _read(*self.items[i])
                for row, tgt in zip(rows, units.encode([r[1] for r in rows])):
                    # CTC needs one frame per label, and one more per repeated label. Use a safe margin.
                    if tgt and len(tgt) <= row[2] * FRAME_RATE - 2:
                        buf.append((row, tgt))
                if len(buf) >= self.buffer_utts:
                    rng.shuffle(buf)
                    head, buf = buf[: self.buffer_utts], buf[self.buffer_utts :]
                    yield from self._batches(head, rng)
            yield from self._batches(buf, rng)
            epoch += 1


class EvalSet:
    """A fixed list of utterances, sorted by duration."""

    def __init__(self, files: list[str], max_utts: int | None = None, seed: int = 0):
        rows = [r for f in files for r in _read(f)]
        if max_utts and len(rows) > max_utts:
            rows = random.Random(seed).sample(rows, max_utts)
        self.rows = sorted(rows, key=lambda r: r[2])
        self.hours = sum(r[2] for r in self.rows) / 3600

    def batches(self, max_batch_seconds: float = 600.0):
        cur = []
        for r in self.rows:
            if cur and r[2] * (len(cur) + 1) > max_batch_seconds:  # rows are sorted, so r is the longest
                yield _collate(cur)
                cur = []
            cur.append(r)
        if cur:
            yield _collate(cur)
