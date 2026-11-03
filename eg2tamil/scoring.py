"""WER and CER as error counts, so the counts add up across batches."""

from dataclasses import dataclass, fields

import jiwer

from .itn import itn
from .text import normalize, normalize_hypothesis


@dataclass
class ErrorCounts:
    word_errors: int = 0
    ref_words: int = 0
    char_errors: int = 0
    ref_chars: int = 0
    utts: int = 0

    def __iadd__(self, other: "ErrorCounts") -> "ErrorCounts":
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def summary(self) -> dict:
        return {
            "wer": self.word_errors / max(self.ref_words, 1),
            "cer": self.char_errors / max(self.ref_chars, 1),
            "utts": self.utts,
            "ref_words": self.ref_words,
        }


def _errors(o) -> tuple[int, int]:
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def score_batch(refs: list[str], hyps: list[str], itn_min: int | None = None, itn_refs: bool = False) -> ErrorCounts:
    """Refs are already normalized. Hyps go through `normalize_hypothesis`. Empty refs are skipped.

    itn_min: change the spoken numbers of at least this value in the hyps to digits (see `itn`), and keep the digits.
    itn_refs: also apply `itn` to the refs. Then the WER does not depend on how a test set writes numbers.
    """
    norm = normalize_hypothesis if itn_min is None else lambda h: normalize(itn(normalize_hypothesis(h), itn_min), strict=False) or ""
    pairs = [(itn(r, itn_min) if itn_refs else r, norm(h)) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return ErrorCounts()
    r, h = zip(*pairs)
    we, rw = _errors(jiwer.process_words(list(r), list(h)))
    ce, rc = _errors(jiwer.process_characters(list(r), list(h)))
    return ErrorCounts(we, rw, ce, rc, len(pairs))
