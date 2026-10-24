"""Tamil text normalization for CTC targets and for ASR hypotheses."""

import unicodedata

# Invisible characters: ZWSP, ZWNJ, ZWJ, soft hyphen, BOM.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍­﻿"))


def _clean(text: str) -> str:
    """Remove invisible characters, change punctuation and symbols to spaces, collapse spaces."""
    text = unicodedata.normalize("NFC", text.translate(_INVISIBLE))
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def _allowed(c: str) -> bool:
    """A space or a Tamil letter or sign. Tamil digits U+0BE6-U+0BEF are not allowed."""
    return c == " " or ("஀" <= c <= "௿" and not "௦" <= c <= "௯")


def normalize(text: str, strict: bool = True) -> str | None:
    """Return the clean text, or None to reject it.

    strict (train): reject empty text and text with a character that is not allowed.
    not strict (valid, test): keep digits and Latin letters (in lowercase), so that a benchmark test set stays
    complete and comparable with published numbers. Reject only empty text.
    """
    text = _clean(text)
    if not strict:
        return text.lower() or None
    return text if text and all(map(_allowed, text)) else None


def normalize_hypothesis(text: str) -> str:
    """Same cleanup as `normalize`, but drop the characters that are not allowed. Never reject."""
    return " ".join("".join(c for c in _clean(text) if _allowed(c)).split())
