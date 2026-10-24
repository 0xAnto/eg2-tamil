"""Run with `python tests/test_model.py`. It downloads EmbeddingGemma 2 on the first run."""

import os
import sys

import numpy as np
import torch
from transformers import AutoFeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eg2tamil.model import READERS, GemmaCTC, LogMel, ModelConfig, greedy_decode, load_encoder  # noqa: E402

ENC = "google/embeddinggemma-2"


def test_log_mel():
    fe, mel = AutoFeatureExtractor.from_pretrained(ENC), LogMel(ENC)
    rng = np.random.default_rng(0)
    waves = [(rng.standard_normal(n) * 0.1).astype("float32") for n in (16000, 40000, 23457)]
    wav = torch.zeros(len(waves), max(map(len, waves)))
    for i, w in enumerate(waves):
        wav[i, : len(w)] = torch.from_numpy(w)
    feats, mask = mel(wav, torch.tensor([len(w) for w in waves]))
    worst = 0.0
    for i, w in enumerate(waves):  # the reference runs one utterance at a time, without padding
        ref = fe([w], sampling_rate=16000, return_tensors="pt")
        n = int(ref["input_features_mask"].sum())
        assert int(mask[i].sum()) == n, (int(mask[i].sum()), n)
        worst = max(worst, (feats[i, :n] - ref["input_features"][0, :n]).abs().max().item())
    print(f"log-mel max abs diff vs AutoFeatureExtractor: {worst:.2e}")
    assert worst < 1e-3


def test_greedy_decode():
    V = 5
    path = [2, 2, 0, 2, 3, 3, 0, 0, 4, 1, 1]  # repeats collapse, a blank splits a repeat, blanks drop
    logits = torch.nn.functional.one_hot(torch.tensor([path, path]), V).float()
    assert greedy_decode(logits, torch.tensor([len(path), 5])) == [[2, 2, 3, 4, 1], [2, 2, 3]]


def test_encoder_truncation():
    """The first K layers of a cut encoder must give the same activations as the first K layers of the full one."""
    enc, _ = load_encoder(ENC, None)
    mel = LogMel(ENC)
    for p in enc.parameters():  # on CPU, bf16 kernels give slightly different numbers if requires_grad differs
        p.requires_grad_(False)
    feats, fmask = mel(torch.randn(2, 16000 * 3) * 0.1, torch.tensor([48000, 30000]))
    acts = []
    hooks = [m.register_forward_hook(lambda mod, i, o: acts.append(o[0] if isinstance(o, tuple) else o))
             for m in [enc.subsample_conv_projection, *enc.layers]]
    with torch.no_grad():
        full = enc(feats.bfloat16(), fmask)
    for h in hooks:
        h.remove()
    assert len(acts) == 13
    model = GemmaCTC(ModelConfig(vocab_size=10, encoder_layers=4, layer_mix=False, **READERS["small"]), enc)
    with torch.no_grad():
        h, mask = model._encode(feats, fmask)
    assert len(model.encoder.layers) == 4 and torch.equal(mask, full.attention_mask)
    assert torch.equal(h, acts[4].float()), (h - acts[4].float()).abs().max()
    print("truncation: 4-layer encoder matches the first 5 activations of the full encoder")


def test_model_frames_and_reader_sizes():
    enc, _ = load_encoder(ENC, None)
    cfg = ModelConfig(vocab_size=10, **READERS["small"])
    model = GemmaCTC(cfg, enc).eval()
    wav = torch.randn(2, 16000 * 4) * 0.1
    lens = torch.tensor([16000 * 4, 16000 * 2])
    with torch.no_grad():
        logits, out_lens = model(wav, lens)
    print(f"4 s -> {out_lens[0].item()} frames, 2 s -> {out_lens[1].item()} frames, logits {tuple(logits.shape)}")
    assert logits.shape == (2, logits.shape[1], 10) and logits.dtype == torch.float32
    assert abs(out_lens[0].item() - 4 * 25) <= 2 and abs(out_lens[1].item() - 2 * 25) <= 2
    assert not any(k.startswith("encoder.") for k in model.trainable_state_dict())
    for name, preset in READERS.items():
        m = GemmaCTC(ModelConfig(vocab_size=3000, **preset), enc)
        print(f"reader {name}: {sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.1f}M trainable params")


if __name__ == "__main__":
    test_log_mel()
    test_greedy_decode()
    test_encoder_truncation()
    test_model_frames_and_reader_sizes()
    print("ok")
