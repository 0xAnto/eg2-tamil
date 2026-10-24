"""Run: python tests/test_lm.py

The beam decoder without a language model must give the same text as greedy decoding.
"""

import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from decode_lm import build_decoder
from eg2tamil.model import greedy_decode
from eg2tamil.units import WB, Units

# A small BPE units object that needs no download.
texts = ["வணக்கம் உலகம்", "வணக்கம் நண்பா", "உலகம் பெரியது"] * 5
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=WB, prepend_scheme="always")
tok.decoder = decoders.Metaspace(replacement=WB, prepend_scheme="always")
tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=60, initial_alphabet=[c for t in texts for c in t if c != " "]))
units = Units(dict(kind="bpe", model_id="none", gemma_ids=[[0]] * tok.get_vocab_size(), tokenizer=tok.to_str()))

# Frames: blanks between units, a repeated frame, and the same unit twice with a blank between.
seq = units.encode(["வணக்கம் உலகம் உலகம்"])[0]
frames = [0]
for u in seq:
    frames += [u, u, 0] if u % 2 else [u, 0, 0]
frames += [seq[-1], 0, seq[-1]]  # the same unit twice, with a blank between
V = len(units)
logp = np.full((len(frames), V), np.log(1e-4), dtype=np.float32)
logp[np.arange(len(frames)), frames] = np.log(1 - 1e-4 * (V - 1))

ref = units.decode(greedy_decode(torch.from_numpy(logp)[None], torch.tensor([len(frames)]))[0])
assert ref.startswith("வணக்கம் உலகம் உலகம்"), ref
assert build_decoder(units).decode(logp, beam_width=8) == ref
print("ok")
