#!/usr/bin/env python
"""Transcribe audio files with a trained eg2-tamil checkpoint. Print one line per file: path, TAB, text.

  python scripts/transcribe.py --ckpt runs/main/best.pt a.wav b.mp3
  python scripts/transcribe.py --ckpt runs/main/best.pt --lm lm/ a.wav    # beam search with a word n-gram LM
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decode_lm import build_decoder  # noqa: E402
from eg2tamil.model import GemmaCTC, greedy_decode  # noqa: E402
from eg2tamil.text import normalize_hypothesis  # noqa: E402
from eg2tamil.units import Units  # noqa: E402

SR = 16000


def split(wav: np.ndarray, lo: float = 20, hi: float = 30, frame: float = 0.05) -> list[np.ndarray]:
    """Cut `wav` into chunks of at most `hi` seconds. Each cut is at the quietest frame between `lo` and `hi` seconds.

    The chunks lie side by side: no gap and no overlap. Audio of `hi` seconds or less stays in one piece.
    """
    # ponytail: energy minimum, no voice detection. A cut can fall inside a word if there is no pause. Upgrade: VAD.
    n, a, chunks = int(frame * SR), 0, []
    while len(wav) - a > hi * SR:
        w = wav[a + int(lo * SR) : a + int(hi * SR)]
        energy = (w[: len(w) // n * n].reshape(-1, n) ** 2).sum(1)
        a_next = a + int(lo * SR) + int(energy.argmin()) * n
        chunks.append(wav[a:a_next])
        a = a_next
    return chunks + [wav[a:]]


def load(path: str) -> np.ndarray:
    """Read any file that libsndfile decodes. Return mono float32 at 16 kHz in [-1, 1]."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    return soxr.resample(x, sr, SR) if sr != SR else x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_seconds", type=float, default=120, help="max padded audio per batch")
    ap.add_argument("--lm", default=None, help="directory with lm.bin and unigrams.txt. Decode with beam search and this LM")
    ap.add_argument("--alpha", type=float, default=0.3, help="LM weight (tuned on valid)")
    ap.add_argument("--beta", type=float, default=0.5, help="word bonus (tuned on valid)")
    ap.add_argument("--unk_offset", type=float, default=-5.0, help="score of a word that is not in unigrams.txt (tuned on valid)")
    ap.add_argument("--beam", type=int, default=64)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, units = GemmaCTC.from_checkpoint(ckpt).to(args.device).eval(), Units(ckpt["units"])
    amp = args.device.startswith("cuda")  # bf16 autocast on the GPU only, as in train.py
    decoder = build_decoder(units, args.lm, args.alpha, args.beta, args.unk_offset) if args.lm else None

    # ponytail: whole files load into memory and the batches are not streamed. Upgrade: stream very long files.
    # One list of chunks for all files. Sort by length, so that a batch holds chunks of similar length.
    chunks = [(i, c) for i, f in enumerate(args.files) for c in split(load(f))]
    order = sorted(range(len(chunks)), key=lambda k: -len(chunks[k][1]))
    texts, k = [""] * len(chunks), 0
    while k < len(order):
        n = max(1, int(args.batch_seconds * SR // len(chunks[order[k]][1])))  # longest chunk comes first
        ids, k = order[k : k + n], k + n
        lens = torch.tensor([len(chunks[j][1]) for j in ids])
        wav = torch.zeros(len(ids), int(lens.max()))
        for r, j in enumerate(ids):
            wav[r, : lens[r]] = torch.from_numpy(chunks[j][1])
        with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            logits, out_lens = model(wav.to(args.device), lens.to(args.device))
        # ponytail: the beam search runs in one process. Upgrade: decoder.decode_batch with a fork Pool, as in decode_lm.py.
        if decoder:  # beam search on the log-probabilities of the valid frames
            logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
            hyps = [decoder.decode(lp[:n], beam_width=args.beam) for lp, n in zip(logp, out_lens.tolist())]
        else:
            hyps = [units.decode(u) for u in greedy_decode(logits, out_lens)]
        for j, h in zip(ids, hyps):
            texts[j] = normalize_hypothesis(h)

    for i, f in enumerate(args.files):
        print(f"{f}\t{' '.join(t for (fi, _), t in zip(chunks, texts) if fi == i and t)}")


if __name__ == "__main__":
    main()
