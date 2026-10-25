"""Run with `python tests/test_export.py`. It needs no download and no real checkpoint."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eg2tamil.model import load_checkpoint  # noqa: E402
from export_hf import export  # noqa: E402


def test_export_roundtrip():
    torch.manual_seed(0)
    ckpt = {
        "config": {"encoder_id": "google/embeddinggemma-2", "vocab_size": 5, "d_model": 8, "layer_mix": True},
        "units": {"kind": "bpe", "model_id": "m", "gemma_ids": [[1], [2, 3], []], "tokenizer": "{\"a\": \"தமிழ்\"}"},
        "step": 123,
        "model": {"layer_logits": torch.randn(13), "blocks.0.qkv.weight": torch.randn(24, 8),
                  "encoder.layers.11.w": torch.randn(4, 4, dtype=torch.bfloat16), "head_bias": torch.zeros(5)},
    }
    with tempfile.TemporaryDirectory() as d:
        export(ckpt, d)
        got = load_checkpoint(d)
    assert got["config"] == ckpt["config"]
    assert got["units"] == ckpt["units"]
    assert got["step"] == ckpt["step"]
    assert got["model"].keys() == ckpt["model"].keys()
    for k, v in ckpt["model"].items():
        assert got["model"][k].dtype == v.dtype and torch.equal(got["model"][k], v), k


if __name__ == "__main__":
    test_export_roundtrip()
    print("ok")
