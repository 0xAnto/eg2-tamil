"""Run: python tests/test_data.py"""

import io
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eg2tamil.text import normalize, normalize_hypothesis

# Normalizer: NFC, invisible characters, punctuation, rejects.
assert normalize("  வணக்கம்,‌  உலகம்! ") == "வணக்கம் உலகம்"
assert normalize("கொ") == "கொ"  # NFC composes the vowel sign
assert normalize("hello") is None and normalize("வணக்கம் 5") is None and normalize("வணக்கம் ௫") is None
assert normalize("...") is None and normalize("வணக்கம் abc") is None
assert normalize_hypothesis("வணக்கம் abc 5, உலகம்") == "வணக்கம் உலகம்"
assert normalize("வணக்கம் ABC 5!", strict=False) == "வணக்கம் abc 5" and normalize("...", strict=False) is None

# Dedup rule: an id that was seen before is dropped. Eval splits come first, so train loses.
from eg2tamil.sources import stem, out_name

assert stem("a/common_voice_ta_1.mp3") == "common_voice_ta_1"
assert out_name("prajwal", "common_voice_ta_1") == "prajwal_cv" and out_name("prajwal", "MICI_1_2") == "prajwal_mile"
assert out_name("mile", "MICI_1_2") == "mile"
import importlib.util

spec = importlib.util.spec_from_file_location("prepare", os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
ids = {("p", "test", "t"): ["A", "B"], ("p", "valid", "v"): ["B", "C", "D", "E", "E"], ("p", "train", "r"): ["D", "F", "F", "C", "G"]}
plan, _ = prepare.assign(ids, lambda name, id_: name, valid_keep=1)
kept = {d: [i for k, p in plan.items() for i, x in zip(ids[k], p) if x == d] for d in ("test", "valid", "train")}
assert kept["test"] == ["A", "B"] and len(kept["valid"]) == 1  # B is in test, so it never goes to valid
assert sorted(kept["train"] + kept["valid"]) == ["C", "D", "E", "F", "G"]  # no id twice, no valid id in train

# FLAC round trip: 16 kHz mono PCM_16 keeps the samples.
x = (np.sin(np.arange(16000) / 10) * 8000).astype(np.int16)
buf = io.BytesIO()
sf.write(buf, x, 16000, format="FLAC", subtype="PCM_16")
y, sr = sf.read(io.BytesIO(buf.getvalue()), dtype="int16")
assert sr == 16000 and np.array_equal(x, y)
print("ok")
