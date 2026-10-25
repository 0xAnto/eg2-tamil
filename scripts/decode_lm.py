#!/usr/bin/env python
"""Score greedy CTC decoding and beam search decoding with a word n-gram LM (KenLM, through pyctcdecode).

  python scripts/decode_lm.py --ckpt runs/main/best.pt --data_dir data/prepared --split test --lm lm/ --alpha 0.3 0.5 --beta 1 2

The first run saves the log-probabilities of each source to runs/logits/<ckpt stem or folder name>_<data_dir name>_<split>/<source>.npy.
--ckpt is a .pt file or a release folder (see export_hf.py).
Later runs read this cache and do not run the model. Delete the directory after a new checkpoint with the same name, or new data in the same data_dir.
--lm is a directory with lm.bin (the KenLM model) and unigrams.txt (one word per line).
Without --lm, the script reports the greedy row only.
The last line is "@@" and one JSON object with all results.
"""

import argparse
import itertools
import json
import os
import struct
import sys
import time
from multiprocessing import get_context

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eg2tamil.data import EvalSet, list_shards  # noqa: E402
from eg2tamil.model import GemmaCTC, greedy_decode, load_checkpoint  # noqa: E402
from eg2tamil.scoring import ErrorCounts, score_batch  # noqa: E402
from eg2tamil.units import Units  # noqa: E402

BLOCK = 128  # utterances per decode call. The block bounds the memory of the float32 copies.


def build_decoder(units, lm=None, alpha=0.3, beta=0.5, unk_offset=-5.0):
    """Return a pyctcdecode beam decoder for `units` (BPE units only).

    lm is a directory with lm.bin and unigrams.txt. Without lm, the decoder has no language model.
    """
    from pyctcdecode import build_ctcdecoder  # only LM decoding needs pyctcdecode

    assert units.kind == "bpe", "the beam decoder needs BPE units"
    # Id 0 is the blank (""), id 1 is <unk> ("⁇" is the unk label of pyctcdecode), id i+2 is BPE piece i.
    labels = ["", "⁇"] + [units._tok.id_to_token(i) for i in range(len(units) - 2)]
    if lm is None:
        return build_ctcdecoder(labels)
    with open(os.path.join(lm, "unigrams.txt"), encoding="utf-8") as f:
        unigrams = [w for w in f.read().split("\n") if w]
    return build_ctcdecoder(labels, os.path.join(lm, "lm.bin"), unigrams, alpha, beta, unk_offset)


def npy_header(shape, size=256):
    """Return an .npy header (fp16, C order) of fixed size, so that the script can write it after the data."""
    d = f"{{'descr': '<f2', 'fortran_order': False, 'shape': {shape}, }}"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", size - 10) + d.ljust(size - 11).encode() + b"\n"


def row_ids(files, es):
    """Return the ids of the rows of `es` in its order (sorted by duration)."""
    t = [pq.read_table(f, columns=["id", "duration"]) for f in files]
    ids = [i for x in t for i in x.column("id").to_pylist()]
    dur = [d for x in t for d in x.column("duration").to_pylist()]
    order = sorted(range(len(ids)), key=lambda i: dur[i])  # stable, as in EvalSet
    assert [dur[i] for i in order] == [r[2] for r in es.rows], "EvalSet and the shards disagree"
    return [ids[i] for i in order]


@torch.no_grad()
def cache_logits(model, es, ids, path, device, amp, batch_seconds, name):
    """Run the model on `es`. Save the fp16 log-softmax of the valid frames, [frames, V], to path (.npy) and path.json.

    Each batch goes to the disk at once, so the memory stays small. The files appear under their final names only at the end.
    """
    model.eval()
    offsets, t0, last, audio = [0], time.time(), 0.0, 0.0
    with open(path + ".tmp", "wb") as f:
        f.write(b"\0" * 256)  # room for the header
        for batch in es.batches(batch_seconds):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits, out_lens = model(batch["wav"].to(device), batch["wav_lens"].to(device))
            logp = torch.log_softmax(logits.float(), -1).half().cpu().numpy()
            for row, n in zip(logp, out_lens.tolist()):
                f.write(row[:n].tobytes())
                offsets.append(offsets[-1] + n)
            audio += batch["wav_lens"].sum().item() / 16000
            if time.time() - last > 30:
                last = time.time()
                print(f"[cache] {name}: {len(offsets) - 1}/{len(es.rows)} utts  {audio / (last - t0):.0f}x realtime", flush=True)
        f.seek(0)
        f.write(npy_header((offsets[-1], logp.shape[-1])))
    with open(path[:-4] + ".json.tmp", "w") as f:
        json.dump({"offsets": offsets, "ids": ids, "refs": [r[1] for r in es.rows]}, f, ensure_ascii=False)
    os.replace(path[:-4] + ".json.tmp", path[:-4] + ".json")
    os.replace(path + ".tmp", path)  # the .npy is the last file: its name means "the cache is complete"
    print(f"[cache] {name}: {len(offsets) - 1} utts done in {time.time() - t0:.0f} s", flush=True)


def blocks(x, offsets):
    """Yield (first, last, [float32 [frames, V] per utterance]) for blocks of BLOCK utterances."""
    for i in range(0, len(offsets) - 1, BLOCK):
        j = min(i + BLOCK, len(offsets) - 1)
        yield i, j, [np.asarray(x[offsets[k] : offsets[k + 1]], dtype=np.float32) for k in range(i, j)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a .pt file or a release folder (see export_hf.py)")
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sources", nargs="*", default=None, help="default: all sources in <data_dir>/<split>")
    ap.add_argument("--lm", default=None, help="directory with lm.bin and unigrams.txt")
    ap.add_argument("--alpha", nargs="+", type=float, default=[0.3], help="LM weight. A list gives a grid with --beta")
    ap.add_argument("--beta", nargs="+", type=float, default=[0.5], help="word bonus. A list gives a grid with --alpha")
    ap.add_argument("--unk_offset", nargs="+", type=float, default=[-5.0], help="score of a word that is not in unigrams.txt. A list adds a grid axis")
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--batch_seconds", type=float, default=600.0, help="padded audio seconds per batch, as in train.py")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    units = Units(ckpt["units"])
    shards = list_shards(args.data_dir, args.split, args.sources)
    assert shards, f"no shards in {args.data_dir}/{args.split}"
    ckpt_name = os.path.basename(os.path.normpath(args.ckpt))  # a folder name, or a file name without .pt
    ckpt_name = ckpt_name if os.path.isdir(args.ckpt) else os.path.splitext(ckpt_name)[0]
    cache = os.path.join("runs", "logits", f"{ckpt_name}_{os.path.basename(os.path.normpath(args.data_dir))}_{args.split}")
    os.makedirs(cache, exist_ok=True)

    # 1. Fill the cache. The model loads only when a source misses.
    model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, files in shards.items():
        path = os.path.join(cache, name + ".npy")
        if os.path.exists(path):
            continue
        if model is None:
            model = GemmaCTC.from_checkpoint(ckpt).to(device)
        es = EvalSet(files)
        cache_logits(model, es, row_ids(files, es), path, device, device.type == "cuda", args.batch_seconds, name)
        del es
    del model

    # 2. Decode from the cache. A row is a setting: greedy, or (alpha, beta).
    data = {}
    for name in shards:
        with open(os.path.join(cache, name + ".json")) as f:
            meta = json.load(f)
        data[name] = (np.load(os.path.join(cache, name + ".npy"), mmap_mode="r"), meta["offsets"], meta["refs"])
    results = {n: {} for n in data}  # {source: {setting: summary}}

    def progress(tag, name, done, total, t0):
        print(f"[{tag}] {name}: {done}/{total} utts  {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    for name, (x, offsets, refs) in data.items():
        counts = ErrorCounts()
        for i, j, arrs in blocks(x, offsets):
            hyps = [units.decode(u) for a in arrs for u in greedy_decode(torch.from_numpy(a)[None], torch.tensor([len(a)]))]
            counts += score_batch(refs[i:j], hyps)
        results[name]["greedy"] = counts.summary()
        progress("greedy", name, len(refs), len(refs), t0)

    if args.lm:
        decoder = build_decoder(units, args.lm)
        for alpha, beta, unk in itertools.product(args.alpha, args.beta, args.unk_offset):
            tag = f"alpha={alpha:g} beta={beta:g} unk={unk:g}"
            decoder.reset_params(alpha=alpha, beta=beta, unk_score_offset=unk)
            t0, last = time.time(), time.time()
            # A new pool for each setting: a forked worker keeps the parameters that it had at the fork.
            with get_context("fork").Pool(args.workers) as pool:
                for name, (x, offsets, refs) in data.items():
                    counts = ErrorCounts()
                    for i, j, arrs in blocks(x, offsets):
                        hyps = decoder.decode_batch(pool, arrs, beam_width=args.beam)
                        counts += score_batch(refs[i:j], hyps)
                        if time.time() - last > 30:
                            last = time.time()
                            progress(tag, name, j, len(refs), t0)
                    results[name][tag] = counts.summary()
            print(f"[{tag}] done in {time.time() - t0:.0f} s", flush=True)

    settings = list(next(iter(results.values())))
    avg = {s: {m: sum(results[n][s][m] for n in results) / len(results) for m in ("wer", "cer")} for s in settings}
    print(f"\n{args.split}, beam {args.beam}. WER and CER in %, lower is better.")
    print(f"{'source':16s} {'setting':32s} {'WER (lower is better)':>22s} {'CER (lower is better)':>22s}")
    for n in results:
        for s in settings:
            print(f"{n:16s} {s:32s} {results[n][s]['wer']:22.2%} {results[n][s]['cer']:22.2%}")
    for s in settings:
        print(f"{'average':16s} {s:32s} {avg[s]['wer']:22.2%} {avg[s]['cer']:22.2%}")
    print("@@" + json.dumps({"mode": "decode_lm", "split": args.split, "beam": args.beam, "results": results, "avg": avg}))


if __name__ == "__main__":
    main()
