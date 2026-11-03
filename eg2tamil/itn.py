"""Inverse text normalization (ITN): change spoken Tamil numbers to digits.

    itn("ஆயிரத்தி தொள்ளாயிரத்தி எழுவத்தாறு வரை")  ->  "1976 வரை"

The model writes numbers as words, often in colloquial forms ("இருவது", "நூற்றி", "முப்பத்த").
A number can be one token or many tokens, and it can end in a suffix ("நூறுகளில்" -> "100 களில்").
Numbers below `min_value` stay as words. "ஒரு" and "ஓர்" alone ("a") always stay as words.
"""

import re

# Tamil vowel sign -> independent vowel. The empty sign is the inherent "அ".
_VOWEL = {"ா": "ஆ", "ி": "இ", "ீ": "ஈ", "ு": "உ", "ூ": "ஊ", "ெ": "எ", "ே": "ஏ", "ை": "ஐ", "ொ": "ஒ", "ோ": "ஓ", "ௌ": "ஔ"}
_SIGNS = "".join(_VOWEL) + "்"

# Each value has its full forms and its stems. A stem ends in a consonant with "்" and takes a link vowel
# ("இருபத்" -> "இருபத்து", "இருபத்தி", "இருபத்தைந்து") or a suffix.
_WORDS = {
    0: ("பூஜ்ஜியம் பூஜ்யம் சைபர்", ""),
    1: ("ஒன்று ஒன்னு ஒண்ணு ஒரு ஓர்", "ஒன்ற் ஒன்ன்"),
    2: ("இரண்டு இரெண்டு ரெண்டு", "இரண்ட் இரெண்ட் ரெண்ட்"),
    3: ("மூன்று மூணு மூனு", "மூன்ற் மூண் மூன்"),
    4: ("நான்கு நாலு", "நான்க் நால்"),
    5: ("ஐந்து அஞ்சு", "ஐந்த் அஞ்ச்"),
    6: ("ஆறு", "ஆற்"),
    7: ("ஏழு", "ஏழ்"),
    8: ("எட்டு", "எட்ட்"),
    9: ("ஒன்பது ஒம்பது", "ஒன்பத் ஒம்பத்"),
    10: ("பத்து", "பத்த்"),
    11: ("பதினொன்று பதினொன்னு பதினொரு", "பதினொன்ற்"),
    12: ("பன்னிரண்டு பன்னிரெண்டு பன்னெண்டு", "பன்னிரண்ட் பன்னிரெண்ட்"),
    13: ("பதின்மூன்று பதிமூணு பதிமூனு", "பதின்மூன்ற்"),
    14: ("பதினான்கு பதினாலு", "பதினான்க்"),
    15: ("பதினைந்து பதினஞ்சு", "பதினைந்த்"),
    16: ("பதினாறு", "பதினாற்"),
    17: ("பதினேழு", "பதினேழ்"),
    18: ("பதினெட்டு", "பதினெட்ட்"),
    19: ("பத்தொன்பது", "பத்தொன்பத்"),
    20: ("இருபது இருவது", "இருபத் இருவத்"),
    30: ("முப்பது", "முப்பத்"),
    40: ("நாற்பது நாப்பது", "நாற்பத் நாப்பத்"),
    50: ("ஐம்பது அம்பது", "ஐம்பத் அம்பத்"),
    60: ("அறுபது அறுவது", "அறுபத் அறுவத்"),
    70: ("எழுபது எழுவது", "எழுபத் எழுவத்"),
    80: ("எண்பது எம்பது", "எண்பத் எம்பத்"),
    90: ("தொண்ணூறு தொன்னூறு", "தொண்ணூற் தொன்னூற்"),
    100: ("நூறு", "நூற் நூத்"),
    200: ("இருநூறு", "இருநூற்"),
    300: ("முந்நூறு முன்னூறு", "முந்நூற் முன்னூற்"),
    400: ("நானூறு நாநூறு", "நானூற் நாநூற்"),
    500: ("ஐநூறு ஐந்நூறு அஞ்சூறு", "ஐநூற் ஐந்நூற்"),
    600: ("அறுநூறு", "அறுநூற்"),
    700: ("எழுநூறு", "எழுநூற்"),
    800: ("எண்ணூறு எண்ணுறு", "எண்ணூற்"),
    900: ("தொள்ளாயிரம்", "தொள்ளாயிரத்"),
    1000: ("ஆயிரம்", "ஆயிரத்"),
    100000: ("லட்சம் இலட்சம்", "லட்சத் இலட்சத்"),
    10**7: ("கோடி", "கோடிய்"),
}
_MULT = (1000, 100000, 10**7)
# A suffix after a number: "நூறுகளில்", "பத்தாம்", "மூன்றில்". Other text after a number word means that the
# token is a different word ("நூற்றாண்டு", century).
_SUFFIXES = ("கள்", "களில்", "களின்", "களுக்கு", "ஆம்", "ஆவது", "இல்", "இன்", "உக்கு", "க்கு", "ஐ", "ஆல்", "ஓடு", "உடன்", "ஆக", "ஆன")


def _glue(stem: str, rest: str) -> str:
    """Join a stem to a word that starts with a vowel: "இரண்ட்" + "ஆயிரம்" = "இரண்டாயிரம்"."""
    sign = {v: s for s, v in _VOWEL.items()}.get(rest[0], "")
    return stem[:-1] + sign + rest[1:]


def _lexicon() -> tuple[dict, dict]:
    full = {w: v for v, (ws, _) in _WORDS.items() for w in ws.split()}
    stems = {s: v for v, (_, ss) in _WORDS.items() for s in ss.split()}
    # A count before a multiplier in one token: "இரண்டாயிரம்", "பத்தாயிரத்து", "ஐயாயிரம்", "மூவாயிரம்".
    for m in _MULT[:1]:  # only ஆயிரம் starts with a vowel and glues to its count
        mf, ms = _WORDS[m][0].split()[0], _WORDS[m][1].split()[0]
        for v in range(1, 100):
            for s in [k for k, x in stems.items() if x == v]:
                full[_glue(s, mf)], stems[_glue(s, ms)] = v * m, v * m
    for w, v in {"ஓராயிரம்": 1000, "மூவாயிரம்": 3000, "ஐயாயிரம்": 5000}.items():
        full[w], stems[w[:-2] + "த்"] = v, v
    return full, stems


_FULL, _STEMS = _lexicon()
_ALL = sorted([(w, v, False) for w, v in _FULL.items()] + [(s, v, True) for s, v in _STEMS.items()], key=lambda t: -len(t[0]))


def _parse(tok: str) -> tuple[list[int], str] | None:
    """Split a token into number values and a suffix: "எழுவத்தாறு" -> ([70, 6], ""). None if it is not a number."""
    for w, v, is_stem in _ALL:
        w = w[:-1] if is_stem else w  # match a stem without its "்"
        if not tok.startswith(w):
            continue
        rest = tok[len(w):]
        if is_stem:
            # A stem is a consonant with "்". It joins in two ways: "இரண்ட்" + "ஆயிரம்" = "இரண்டாயிரம்" (the "்"
            # goes), or "இருபத்" + "ஐந்து" = "இருபத்தைந்து" (the consonant repeats).
            c = w[-1]
            if rest.startswith("்" + c):
                rest = rest[2:]
                rest = (_VOWEL[rest[0]] + rest[1:]) if rest[:1] in _VOWEL else "அ" + rest
            elif rest[:1] in _VOWEL:
                rest = _VOWEL[rest[0]] + rest[1:]
            elif rest != "்":
                continue
            else:
                rest = ""
            if rest in ("", "அ", "இ", "உ"):  # no link vowel, or a link vowel only: "ஆயிரத்தி", "முப்பத்த"
                return [v], ""
            if rest[0] in "அஇஉ" and (t := _parse(rest[1:])):  # a link vowel, then a number: "இருபத்துமூன்று"
                return [v] + t[0], t[1]
        elif rest and rest[0] in _SIGNS:
            continue  # a longer word that only starts like a number word
        if not rest:
            return [v], ""
        if t := _parse(rest):
            return [v] + t[0], t[1]
        if rest.startswith(_SUFFIXES):
            return [v], rest
    return None


def _value(vals: list[int]) -> int | None:
    """Add the place values: [1000, 900, 70, 6] -> 1976. None if the order is not a number ("இரண்டு மூன்று")."""
    total, cur, cap, big = 0, 0, None, None  # cap: the next value must be below it. big: the last multiplier
    for v in vals:
        m = next((x for x in reversed(_MULT) if v >= x and v % x == 0), None)
        if m:  # a multiplier, or a count and a multiplier in one token ("இரண்டாயிரம்")
            if (cur and v != m) or (big and m >= big):
                return None
            total, cur, cap, big = total + (cur or 1) * v, 0, m, m
        else:
            if (cap and v >= cap) or (v == 0 and (total or cur)):
                return None
            cur += v
            cap = 100 if v >= 100 else 10 if v >= 20 and v % 10 == 0 else 1
    return total + cur


def itn(text: str, min_value: int = 10) -> str:
    """Change each spoken number of at least `min_value` to digits. Other words stay as they are."""
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        best = None  # (end, value, suffix): the longest run of tokens that is one number
        vals = []
        for j in range(i, len(toks)):
            p = _parse(toks[j])
            if not p:
                break
            vals = vals + p[0]
            n = _value(vals)
            if n is None:
                break
            best = (j + 1, n, p[1])
            if p[1]:
                break  # a suffix ends the number
        if best and best[1] >= min_value and not (best[0] == i + 1 and toks[i] in ("ஒரு", "ஓர்")):
            out += [str(best[1])] + ([best[2]] if best[2] else [])  # ponytail: the suffix is a separate word, as in most refs
            i = best[0]
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)

