"""Download the raw datasets and write clean 16 kHz FLAC shards (columns: id, audio, text, duration).

python scripts/prepare.py --sources prajwal mile --raw_dir data/raw --out_dir data/prepared

Steps: (1) read the id columns from the Hub, (2) decide for each row: train, valid, test or drop,
(3) download only the files that have a kept row, (4) convert the files in parallel, (5) write stats and units.
"""

import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eg2tamil import sources
from eg2tamil.data import FRAME_RATE, list_shards
from eg2tamil.text import normalize
from eg2tamil.units import UNK, Units

SR, MIN_S, MAX_S, ROW_GROUP = 16_000, 0.5, 30.0, 256
MAX_EVAL_S = 120.0  # the encoder takes up to about 5 min; long eval clips cost only memory
SCHEMA = pa.schema([("id", pa.string()), ("audio", pa.binary()), ("text", pa.string()), ("duration", pa.float32())])
GEMMA = "google/embeddinggemma-2"


def assign(ids, out, valid_keep=2000, seed=0):
    """Decide the destination of every row.

    ids: {(source, split, file): [id, ...]} in priority order. out(source, id) gives the output source name.
    Return ({key: [destination or None per row]}, overlap counts). None means drop.
    The eval ids are all test ids and the kept valid ids. They never go to train.
    An id that was seen before is dropped, so no clip appears twice.
    """
    first, plan, counts = {}, {k: [None] * len(v) for k, v in ids.items()}, Counter()

    def claim(key, k, id_, dest):
        if id_ in first:
            counts[f"{key[1]}/{out(key[0], id_)} already in {first[id_]}"] += 1
        else:
            first[id_], plan[key][k] = dest, dest

    for key, rows in ids.items():  # 1. test rows
        if key[1] == "test":
            for k, id_ in enumerate(rows):
                claim(key, k, id_, "test")
    cand = {}  # 2. valid rows: keep a fixed random sample per output source. The rest goes to train in step 3.
    for key, rows in ids.items():
        if key[1] == "valid":
            for k, id_ in enumerate(rows):
                if id_ not in first and id_ not in cand:
                    cand[id_] = (key, k)
    by_src = {}
    for id_, (key, k) in cand.items():
        by_src.setdefault(out(key[0], id_), []).append((id_, key, k))
    for src in sorted(by_src):
        for id_, key, k in random.Random(f"{seed}:{src}").sample(by_src[src], min(valid_keep, len(by_src[src]))):
            first[id_], plan[key][k] = "valid", "valid"
    for name in dict.fromkeys(k[0] for k in ids):  # 3. train rows and the valid rest, by source order
        for split in ("valid", "train"):
            for key, rows in ids.items():
                if key[:2] == (name, split):
                    for k, id_ in enumerate(rows):
                        if plan[key][k] is None:
                            claim(key, k, id_, "train")
    return plan, counts


def reject_reason(raw):
    """Name the reason why `normalize` rejects `raw`."""
    if re.search("[A-Za-z]", raw):
        return "latin"
    if re.search(r"\d", raw):
        return "digit"
    return "empty" if not re.search(r"\w", raw) else "other_script"


def process_file(job):
    """Convert one input file. Return its stats. The marker file means the file is done."""
    name, rel, plan, raw_dir, out_dir = job
    spec, base = sources.SOURCES[name], sources.stem(rel.replace("/", "_"))  # unique, because FLEURS files share a name
    marker = os.path.join(out_dir, ".done", f"{name}.{base}.json")
    if os.path.exists(marker):
        return json.load(open(marker))
    t0 = time.time()
    drops, rows, secs, rates, bad_chars = Counter(), Counter(), Counter(), Counter(), Counter()
    writers, bufs, n_in = {}, {}, 0

    def flush(key, force=False):
        if len(bufs[key]) >= ROW_GROUP or (force and bufs[key]):
            if key not in writers:
                d = os.path.join(out_dir, *key)
                os.makedirs(d, exist_ok=True)
                writers[key] = pq.ParquetWriter(os.path.join(d, base + ".parquet.tmp"), SCHEMA)
            writers[key].write_table(pa.table(dict(zip(SCHEMA.names, zip(*bufs[key]))), schema=SCHEMA))
            bufs[key] = []

    pf = pq.ParquetFile(os.path.join(raw_dir, name, rel))
    for batch in pf.iter_batches(ROW_GROUP, columns=[spec["id"], spec["text"], "audio"]):
        cols = batch.to_pydict()
        for id_, raw, aud in zip(cols[spec["id"]], cols[spec["text"]], cols["audio"]):
            if n_in >= len(plan):  # the plan covers max_rows_per_file rows at most
                break
            n_in += 1
            dest = plan[n_in - 1]
            if dest is None:
                continue
            id_, raw = sources.stem(id_), raw or ""
            text = normalize(raw, strict=dest == "train")  # eval splits keep digits and Latin letters
            if text is None:
                drops[f"{name}/text_" + reject_reason(raw)] += 1
                bad_chars.update(c for c in raw if not re.match(r"[஀-௿\s\W\d]", c))
                continue
            try:
                x, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                rates[sr] += 1
                if sr != SR:
                    x = soxr.resample(x, sr, SR)
                flac = io.BytesIO()
                sf.write(flac, x, SR, format="FLAC", subtype="PCM_16")
            except Exception:
                drops[f"{name}/decode_error"] += 1
                continue
            dur = len(x) / SR
            if not MIN_S <= dur <= (MAX_S if dest == "train" else MAX_EVAL_S):  # eval sets stay complete
                drops[f"{name}/" + ("too_short" if dur < MIN_S else "too_long")] += 1
                continue
            key = (dest, sources.out_name(name, id_))
            bufs.setdefault(key, []).append((id_, flac.getvalue(), text, dur))
            rows["/".join(key)] += 1
            secs["/".join(key)] += dur
            flush(key)
        if n_in >= len(plan):
            break
    for key in bufs:
        flush(key, force=True)
    for key, w in writers.items():
        w.close()
        final = os.path.join(out_dir, *key, base + ".parquet")
        os.replace(final + ".tmp", final)
    stats = dict(source=name, kept_in=sum(map(bool, plan)), rows=rows, seconds=secs, drops=drops, bad_chars=dict(bad_chars.most_common(30)),
                 sample_rates={str(k): v for k, v in rates.items()}, input_rows=n_in, wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker + ".tmp", "w") as f:
        json.dump(stats, f)
    os.replace(marker + ".tmp", marker)
    return stats


def read_texts(out_dir, split):
    """Return {source: (texts, durations)} for one split of the prepared data."""
    res = {}
    for src, files in list_shards(out_dir, split).items():
        t = pa.concat_tables([pq.read_table(f, columns=["text", "duration"]) for f in files])
        res[src] = (t.column("text").to_pylist(), t.column("duration").to_pylist())
    return res


def unit_stats(units, texts, durs):
    """UNK rate, units per word, units per second, and the share of utterances that fail the CTC length rule."""
    n = np.array([len(e) for e in units.encode(texts)])
    unk = sum(e.count(UNK) for e in units.encode(texts))
    dur = np.array(durs)
    rate = n / dur
    return dict(utts=len(texts), unk_rate=unk / max(n.sum(), 1), units_per_word=n.sum() / sum(len(t.split()) for t in texts),
                units_per_s_p50=np.percentile(rate, 50), units_per_s_p99=np.percentile(rate, 99), units_per_s_max=rate.max(),
                encoder_frames_per_s=FRAME_RATE, ctc_fail_share=np.mean((n == 0) | (n > dur * FRAME_RATE - 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["prajwal", "mile"])
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--valid_keep", type=int, default=2000, help="valid utts per output source. The rest goes to train.")
    ap.add_argument("--bpe_vocab", type=int, default=1024)
    ap.add_argument("--dry_run", action="store_true", help="scan the ids and print the download plan only")
    ap.add_argument("--max_files_per_split", type=int, help="smoke test only")
    ap.add_argument("--max_rows_per_file", type=int, help="smoke test only")
    a = ap.parse_args()

    # Pass 1: read only the id columns from the Hub (or from disk, if the file is already there).
    keys = [(n, s, f) for s in ("test", "valid", "train") for n in a.sources if s in sources.SOURCES[n]["splits"]
            for f in sources.remote_files(n, s, a.max_files_per_split)]
    with ThreadPoolExecutor(2) as ex:
        scanned = list(ex.map(lambda k: sources.read_ids(k[0], k[2], a.raw_dir, a.max_rows_per_file), keys))
    ids = {k: v[0] for k, v in zip(keys, scanned)}
    size = {k: v[1] for k, v in zip(keys, scanned)}
    plan, overlaps = assign(ids, sources.out_name, a.valid_keep)
    need = [k for k in keys if any(plan[k])]
    gb = lambda ks: sum(size[k] for k in ks) / 1e9
    print(f"overlaps: {dict(overlaps)}\ndownload: {len(need)} of {len(keys)} files, {gb(need):.1f} of {gb(keys):.1f} GB"
          f" (saves {gb(keys) - gb(need):.1f} GB)")
    print("skipped files:", [f"{k[0]}:{os.path.basename(k[2])}" for k in keys if k not in need])
    if a.dry_run:
        return

    for n in a.sources:
        sources.download(n, a.raw_dir, [k[2] for k in need if k[0] == n])

    # Pass 2: decode, normalize, resample, encode. One task per input file.
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(process_file, [(n, f, plan[(n, s, f)], a.raw_dir, a.out_dir) for n, s, f in need]))

    stats = dict(hours=Counter(), rows=Counter(), drops=Counter(), overlaps=dict(overlaps), bad_chars=Counter(),
                 sample_rates=Counter(), input_rows=0, worker_s=0.0, audio_s=0.0)
    for r in results:
        stats["rows"].update(r["rows"])
        stats["hours"].update({k: v / 3600 for k, v in r["seconds"].items()})
        stats["audio_s"] += sum(r["seconds"].values())
        for key in ("drops", "bad_chars", "sample_rates"):
            stats[key].update(r[key])
        stats["input_rows"] += r["input_rows"]
        stats["worker_s"] += r["wall_s"]
    # Reject rate = rows dropped by the text rule or by decoding or by duration, per input source.
    for n in a.sources:
        kept_in = sum(r["kept_in"] for r in results if r["source"] == n)
        stats.setdefault("reject_rate", {})[n] = sum(v for k, v in stats["drops"].items() if k.startswith(n + "/")) / max(kept_in, 1)
        stats.setdefault("latin_rate", {})[n] = stats["drops"][f"{n}/text_latin"] / max(kept_in, 1)
    stats["x_realtime_per_worker"] = stats["audio_s"] / max(stats["worker_s"], 1e-9)
    stats["bad_chars"] = dict(stats["bad_chars"].most_common(30))

    # Honest eval: the share of eval utterances whose text also occurs in train (not dropped, only reported).
    train = read_texts(a.out_dir, "train")
    train_texts = {t for texts, _ in train.values() for t in texts}
    stats["eval_text_seen_in_train"] = {
        f"{sp}/{src}": float(np.mean([t in train_texts for t in texts]))
        for sp in ("valid", "test") for src, (texts, _) in read_texts(a.out_dir, sp).items()}

    # Units: both kinds, from the train texts.
    texts = [t for ts, _ in train.values() for t in ts]
    durs = [d for _, ds in train.values() for d in ds]
    test = read_texts(a.out_dir, "test")
    test_texts, test_durs = [t for ts, _ in test.values() for t in ts], [d for _, ds in test.values() for d in ds]
    stats["units"] = {}
    for kind in ("gemma", "bpe"):
        units, build = Units.build(GEMMA, texts, kind=kind, vocab=a.bpe_vocab)
        units.save(os.path.join(a.out_dir, f"units_{kind}.json"))
        stats["units"][kind] = dict(count=len(units), **build, train=unit_stats(units, texts, durs),
                                    test=unit_stats(units, test_texts, test_durs))
    with open(os.path.join(a.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
