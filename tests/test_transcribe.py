"""Run: python tests/test_transcribe.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from transcribe import SR, split

rng = np.random.default_rng(0)
wav = rng.normal(size=70 * SR).astype(np.float32)

# Short audio stays in one piece.
assert len(split(wav[: 30 * SR])) == 1 and len(split(wav[: 5 * SR])) == 1

# Long audio: every chunk has at most 30 s. The chunks cover the signal with no gap and no overlap.
chunks = split(wav)
assert len(chunks) == 3 and all(len(c) <= 30 * SR for c in chunks)
assert np.array_equal(np.concatenate(chunks), wav)

# The cut falls in the silent region (24.0 s to 24.5 s) and not at 20 s or 30 s.
wav[int(24 * SR) : int(24.5 * SR)] *= 0.001
cut = len(split(wav)[0])
assert 24 * SR <= cut <= 24.5 * SR, cut / SR
print("ok")
