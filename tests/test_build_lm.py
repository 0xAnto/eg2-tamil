"""Run: python tests/test_build_lm.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from build_lm import clean_doc, ngrams

# Split at line breaks and at . ? ! । . Normalize. Count rejects (Latin, digits) and exclusions. Skip blank pieces.
doc = "வணக்கம் உலகம்! இது ஒரு சோதனை.\nHello world?  \nஎன் எண் 5 ஆகும்।\nஇது ஒரு சோதனை. சோதனை, சோதனை"
kept, rej, exc = clean_doc(doc, {"இது ஒரு சோதனை"}, set())
assert kept == ["வணக்கம் உலகம்", "சோதனை சோதனை"], kept
assert (rej, exc) == (2, 2), (rej, exc)
assert clean_doc("", set(), set()) == ([], 0, 0)
# A transcript that equals a reference is excluded as a whole, also when a "." splits it.
assert clean_doc("இது ஒரு. சோதனை", {"இது ஒரு சோதனை"}, set()) == ([], 0, 1)
# A sentence that holds a 5-gram of a reference is excluded. A sentence with only 4 of its words is kept.
ref = "அ ஆ இ ஈ உ ஊ"
kept, rej, exc = clean_doc("க அ ஆ இ ஈ உ ங. அ ஆ இ ஈ ச", {ref}, ngrams(ref))
assert (kept, exc) == (["அ ஆ இ ஈ ச"], 1), (kept, exc)
print("ok")
