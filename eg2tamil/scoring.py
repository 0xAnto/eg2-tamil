"""WER and CER as error counts, so the counts add up across batches."""

from dataclasses import dataclass, fields

import jiwer

from .text import normalize_hypothesis


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


def score_batch(refs: list[str], hyps: list[str]) -> ErrorCounts:
    """Refs are already normalized. Hyps go through `normalize_hypothesis`. Empty refs are skipped."""
    pairs = [(r, normalize_hypothesis(h)) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return ErrorCounts()
    r, h = zip(*pairs)
    we, rw = _errors(jiwer.process_words(list(r), list(h)))
    ce, rc = _errors(jiwer.process_characters(list(r), list(h)))
    return ErrorCounts(we, rw, ce, rc, len(pairs))
