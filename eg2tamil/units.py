"""CTC output units. Two kinds:

gemma: the Gemma tokenizer tokens that occur in the Tamil train text.
bpe:   a small BPE vocabulary trained on the Tamil train text.

Unit ids: 0 = CTC blank, 1 = <unk>, 2.. = the kept tokens (gemma: most frequent first, bpe: vocabulary order).
`gemma_ids[i]` lists the Gemma token ids of unit i+2. The model starts each head row from their mean embedding.
"""

import json
import re
import unicodedata
from collections import Counter

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

BLANK, UNK = 0, 1
WB = "▁"  # word boundary mark


class Units:
    def __init__(self, d: dict):
        self.kind, self.model_id, self.gemma_ids = d["kind"], d["model_id"], d["gemma_ids"]
        self._d = d
        if self.kind == "gemma":
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._to_unit = {g[0]: i + 2 for i, g in enumerate(self.gemma_ids)}
        else:
            self._tok = Tokenizer.from_str(d["tokenizer"])

    def __len__(self) -> int:
        return len(self.gemma_ids) + 2

    def encode(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "bpe":
            return [[i + 2 for i in e.ids] for e in self._tok.encode_batch(texts)]
        # The leading space gives the first word the same "▁word" token as the other words.
        ids = self._tok([" " + t for t in texts], add_special_tokens=False).input_ids
        return [[self._to_unit.get(g, UNK) for g in x] for x in ids]

    def decode(self, unit_ids: list[int]) -> str:
        if self.kind == "bpe":
            return self._tok.decode([u - 2 for u in unit_ids if u >= 2])
        return self._tok.decode([self.gemma_ids[u - 2][0] for u in unit_ids if u >= 2]).strip()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._d, f)

    @classmethod
    def load(cls, path: str) -> "Units":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, model_id: str, texts, kind: str = "gemma", min_count: int = 2, vocab: int = 1024) -> tuple["Units", dict]:
        """Build units from the train texts. `min_count` is for gemma, `vocab` is for bpe."""
        texts = list(texts)
        gemma = AutoTokenizer.from_pretrained(model_id)
        words = sum(len(t.split()) for t in texts)
        if kind == "bpe":
            # ponytail: BPE merges code points, not Tamil letter clusters. Upgrade: pre-split into grapheme clusters.
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=WB, prepend_scheme="always")
            tok.decoder = decoders.Metaspace(replacement=WB, prepend_scheme="always")
            # Every assigned Tamil code point is in the alphabet, so every normalized word decomposes.
            alpha = [c for c in map(chr, range(0x0B80, 0x0C00)) if unicodedata.name(c, None) and not c.isdigit()]
            tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=alpha, show_progress=False))
            pieces = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
            gemma_ids = gemma([p.replace(WB, " ") for p in pieces], add_special_tokens=False).input_ids
            d = dict(kind=kind, model_id=model_id, gemma_ids=gemma_ids, tokenizer=tok.to_str())
            return cls(d), {"vocab": len(pieces), "gemma_tokens_per_unit": sum(map(len, gemma_ids)) / len(pieces)}
        counts: Counter = Counter()
        for i in range(0, len(texts), 10_000):  # batches keep the memory small
            for ids in gemma([" " + t for t in texts[i : i + 10_000]], add_special_tokens=False).input_ids:
                counts.update(ids)
        kept = [g for g, c in counts.most_common() if c >= min_count]
        total = sum(counts.values())
        byte_toks = sum(c for g, c in counts.items() if re.fullmatch(r"<0x[0-9A-F]{2}>", gemma.convert_ids_to_tokens(g)))
        stats = {
            "distinct_tokens": len(counts),
            "kept_tokens": len(kept),
            "total_tokens": total,
            "token_coverage": sum(counts[g] for g in kept) / max(total, 1),
            "byte_fallback_fraction": byte_toks / max(total, 1),
            "tokens_per_word": total / max(words, 1),
        }
        return cls(dict(kind=kind, model_id=model_id, gemma_ids=[[g] for g in kept])), stats
