"""A small CTC reader on top of the audio encoder of EmbeddingGemma 2.

    wav -> log-mel (GPU) -> Gemma audio encoder (frozen conformer, 25 frames/s)
        -> learned softmax mix of the 13 encoder activations
        -> LayerNorm + Linear -> depthwise conv -> N pre-norm transformer blocks (the reader)
        -> head over the units. The head rows start as Gemma token embeddings.
        -> CTC

The idea comes from qnguyen3/gemma-asr (Vietnamese). This file is a new implementation.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

READERS = {  # reader size presets: d_model, n_layers, n_heads
    "small": dict(d_model=256, n_layers=4, n_heads=4),
    "base": dict(d_model=512, n_layers=6, n_heads=8),
    "large": dict(d_model=768, n_layers=8, n_heads=12),
}


@dataclass
class ModelConfig:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0  # units including blank and unk
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    layer_mix: bool = True  # False: use the last conformer layer only
    encoder_layers: int = 12  # run only the first K conformer layers (and mix K+1 activations)
    unfreeze_top_k: int = 0  # number of top encoder layers to fine-tune. They count from layer K.
    head_init: str = "text"  # "text": Gemma token embeddings, or "random"


class LogMel(nn.Module):
    """Batched log-mel. It gives the same numbers as `Gemma4AudioFeatureExtractor`, which is a numpy loop."""

    def __init__(self, encoder_id: str):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(encoder_id)
        assert fe.preemphasis == 0 and fe.dither == 0 and fe.input_scale_factor == 1, "unsupported extractor"
        assert fe.per_bin_mean is None and fe.per_bin_stddev is None, "unsupported extractor"
        self.frame, self.hop, self.n_fft, self.floor = fe.frame_length, fe.hop_length, fe.fft_length, float(fe.mel_floor)
        self.register_buffer("window", torch.tensor(fe.window, dtype=torch.float32), persistent=False)
        self.register_buffer("filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        """wav (B, L) float in [-1, 1], zero padded. Returns features (B, T, 128) and a bool mask (B, T)."""
        with torch.autocast(wav.device.type, enabled=False):  # keep fp32: bf16 would spoil the mel
            left = self.frame // 2
            x = F.pad(wav.float(), (left, 0))
            frames = x.unfold(-1, self.frame + 1, self.hop)[..., :-1]  # (B, T, frame)
            mag = torch.fft.rfft(frames * self.window, n=self.n_fft).abs()
            feats = torch.log(mag @ self.filters + self.floor)
            end = torch.arange(feats.shape[1], device=wav.device) * self.hop + self.frame
            mask = end[None] < wav_lens[:, None] + left
            return feats * mask[..., None], mask


class Block(nn.Module):
    """Pre-norm transformer block. Attention is bidirectional and ignores padded keys."""

    def __init__(self, d: int, heads: int, ff_mult: int, p: float):
        super().__init__()
        self.heads, self.p = heads, p
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(p), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.norm2(x)))


def load_encoder(encoder_id: str, unit_gemma_ids: list[int] | None):
    """Return (audio_tower, text embeddings of the units or None). The rest of EmbeddingGemma 2 is dropped.

    `unit_gemma_ids[i]` is the list of Gemma token ids of unit i. The unit embedding is the mean of their embeddings.
    """
    full = AutoModel.from_pretrained(encoder_id, dtype=torch.bfloat16, vision_config=None)
    emb = None
    if unit_gemma_ids is not None:
        table = full.language_model.embed_tokens.weight.detach()
        # A unit with no Gemma tokens gets a zero row; F.normalize keeps it at zero.
        emb = torch.stack([table[ids].float().mean(0) if ids else table.new_zeros(table.shape[1]).float() for ids in unit_gemma_ids])
    encoder = full.audio_tower
    del full
    return encoder, emb


def _span_mask(n_masks: int, max_w, hi, length: int, B: int, device) -> torch.Tensor:
    """Random spans along one axis. Span width is up to `max_w`, start is below `hi` (both (B,1) or int)."""
    idx = torch.arange(length, device=device)
    drop = torch.zeros(B, length, dtype=torch.bool, device=device)
    for _ in range(n_masks):
        w = (torch.rand(B, 1, device=device) * (max_w + 1)).long()
        s = (torch.rand(B, 1, device=device) * (hi - w).clamp(min=1)).long()
        drop |= (idx >= s) & (idx < s + w)
    return drop


class GemmaCTC(nn.Module):
    def __init__(self, cfg: ModelConfig, encoder: nn.Module, unit_emb: torch.Tensor | None = None):
        super().__init__()
        self.cfg, self.encoder, self.mel = cfg, encoder, LogMel(cfg.encoder_id)
        n_enc, enc_dim, d = cfg.encoder_layers, encoder.config.hidden_size, cfg.d_model
        assert 1 <= n_enc <= len(encoder.layers) and cfg.unfreeze_top_k <= n_enc, "bad encoder_layers"
        # The HF forward loops over `self.layers[:config.num_hidden_layers]`, so cutting the list is enough.
        # `output_proj` is never used here, so it becomes a no-op.
        encoder.layers = encoder.layers[:n_enc]
        encoder.config.num_hidden_layers = n_enc
        encoder.output_proj = nn.Identity()

        for p in encoder.parameters():
            p.requires_grad_(False)
        if cfg.unfreeze_top_k > 0:  # fp32 master weights. The forward pass runs under bf16 autocast.
            encoder.float()
            for p in encoder.layers[n_enc - cfg.unfreeze_top_k :].parameters():
                p.requires_grad_(True)

        # `output_hidden_states` swaps the last raw layer for the 1536-d output_proj, so hooks collect
        # 13 activations: the subsampler output and the 12 conformer layers. The subsampler returns (hidden, mask).
        self._acts: list[torch.Tensor] = []
        for m in [encoder.subsample_conv_projection, *encoder.layers]:
            m.register_forward_hook(lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out))

        self.layer_logits = nn.Parameter(torch.zeros(n_enc + 1)) if cfg.layer_mix else None
        self.in_norm, self.in_proj = nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=d)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(d)

        # The head works in the text-embedding space of Gemma (512-d), so unit rows can start as token embeddings.
        text_dim = 512 if unit_emb is None else unit_emb.shape[1]
        self.to_text = nn.Linear(d, text_dim)
        if cfg.head_init == "text" and unit_emb is not None:
            rows = F.normalize(unit_emb, dim=-1)
        else:
            rows = torch.randn(cfg.vocab_size - 2, text_dim) * text_dim**-0.5
        self.unit_rows = nn.Parameter(rows)
        self.special_rows = nn.Parameter(torch.randn(2, text_dim) * text_dim**-0.5)  # blank, unk
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # the encoder has no dropout, even when we fine-tune its top layers
        return self

    def _encode(self, feats: torch.Tensor, feat_mask: torch.Tensor):
        self._acts.clear()
        # ponytail: no_grad, not inference_mode. Inference tensors cannot go into the autograd graph of the layer mix.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.unfreeze_top_k > 0):
            out = self.encoder(feats.to(next(self.encoder.parameters()).dtype), feat_mask)
        acts, self._acts = self._acts, []
        assert len(acts) == len(self.encoder.layers) + 1, "hooks did not fire once per module"
        if self.layer_logits is None:
            return acts[-1].float(), out.attention_mask
        w = torch.softmax(self.layer_logits.float(), 0)
        return sum(w[i] * a.float() for i, a in enumerate(acts)), out.attention_mask

    def spec_augment(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """2 frequency masks (width up to 27 bins) and 10 time masks (width up to 4% of the length)."""
        B, T, n_mel = feats.shape
        lens = mask.sum(1, keepdim=True)
        fill = (feats * mask[..., None]).sum(1, keepdim=True) / lens.clamp(min=1)[..., None]  # mean per utterance
        drop = _span_mask(2, 27, n_mel, n_mel, B, feats.device)[:, None, :]
        drop = drop | _span_mask(10, (lens * 0.04).long().clamp(min=1), lens, T, B, feats.device)[:, :, None]
        return torch.where(drop, fill, feats) * mask[..., None]

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor, specaug: bool = False):
        """Return CTC logits (B, T, V) in float32 and the output lengths (B,)."""
        feats, feat_mask = self.mel(wav, wav_lens)
        if specaug:
            feats = self.spec_augment(feats, feat_mask)
        h, mask = self._encode(feats, feat_mask)
        x = self.in_proj(self.in_norm(h))
        x = x + F.gelu(self.pos_conv((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))
        for blk in self.blocks:
            x = blk(x, mask)
        t = self.to_text(self.out_norm(x))
        with torch.autocast(t.device.type, enabled=False):  # fp32 logits for the CTC loss
            rows = torch.cat([self.special_rows, self.unit_rows]).float()
            return t.float() @ rows.t() + self.head_bias.float(), mask.sum(1)

    def trainable_state_dict(self) -> dict:
        """All weights except the frozen encoder weights. Those come from the Hugging Face checkpoint."""
        train = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.") or k in train}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GemmaCTC":
        cfg = ModelConfig(**ckpt["config"])
        model = cls(cfg, load_encoder(cfg.encoder_id, None)[0])
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"bad checkpoint: missing={missing} unexpected={unexpected}")
        return model

    def config_dict(self) -> dict:
        return asdict(self.cfg)


@torch.no_grad()
def greedy_decode(logits: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    """Best path: argmax, collapse repeats, drop blanks (id 0)."""
    best, out = logits.argmax(-1).cpu(), []
    for seq, n in zip(best, lens.tolist()):
        seq = seq[:n]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != 0].tolist())
    return out
