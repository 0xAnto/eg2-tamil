"""Build a Tamil word n-gram language model (KenLM) for CTC beam search.

Text sources: the train transcripts of prajwal and fleurs, Tamil Wikipedia, and a part of ai4bharat/sangraha.
Output in --out_dir: lm.bin (KenLM binary), unigrams.txt (one word per line, most frequent first), text.txt (the LM text), stats.json.
The last line of the run starts with "@@" and holds one JSON object with the stats.

    python scripts/build_lm.py --out_dir data/lm
"""

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eg2tamil.data import list_shards
from eg2tamil.sources import SOURCES, remote_files
from eg2tamil.text import normalize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KENLM = os.path.join(ROOT, "tools", "kenlm", "build", "bin")
_SPLIT = re.compile(r"[\r\n.?!।‥…]+")  # line breaks and sentence-end punctuation
N_GRAM = 5  # an LM sentence that shares a word 5-gram with a reference is excluded


def ngrams(text: str) -> set[str]:
    w = text.split()
    return {" ".join(w[i : i + N_GRAM]) for i in range(len(w) - N_GRAM + 1)}


def clean_doc(doc: str, exclude: set[str], grams: set[str]) -> tuple[list[str], int, int]:
    """Split one document into sentences. Normalize each one with the train rule (strict).
    Return (kept sentences, number of rejected sentences, number of excluded sentences).
    A sentence is excluded when it equals a reference text of valid or test, or when it shares a word 5-gram (`grams`)
    with a reference. The 5-gram rule removes a web sentence that holds a reference inside it.
    A document that equals a reference is excluded as a whole, so that a transcript with a "." or a "?" inside it
    cannot go into the LM in parts."""
    if normalize(doc, strict=False) in exclude:  # the references are stored with strict=False
        return [], 0, 1
    kept, rejected, excluded = [], 0, 0
    for s in _SPLIT.split(doc):
        if not s.strip():
            continue
        s = normalize(s, strict=True)
        if s is None:
            rejected += 1
        elif s in exclude or not grams.isdisjoint(ngrams(s)):
            excluded += 1
        else:
            kept.append(s)
    return kept, rejected, excluded


def read_text(fs: HfFileSystem, path: str, col: str, limit: int | None) -> list[str]:
    """Read one text column of a remote parquet file, one row group at a time, until limit rows.
    cache_type none fetches only the bytes that pyarrow asks for, so the audio columns stay on the Hub."""
    pf = pq.ParquetFile(fs.open(path, cache_type="none"))
    out = []
    for g in range(pf.num_row_groups):
        out += pf.read_row_group(g, columns=[col]).column(0).to_pylist()
        if limit and len(out) >= limit:
            break
    return out[:limit]


def doc_sources(sangraha_files: int) -> dict:
    """Return {source name: (file paths for HfFileSystem, text column)}, in the order of use."""
    def hub(repo, files, revision=None):
        return [f"datasets/{repo}" + (f"@{quote(revision, safe='')}" if revision else "") + "/" + f for f in files]

    def listing(repo, prefix):
        return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))

    out = {n: (hub(SOURCES[n]["repo"], remote_files(n, "train"), SOURCES[n].get("revision")), SOURCES[n]["text"]) for n in ("prajwal", "fleurs")}
    out["wikipedia"] = (hub("wikimedia/wikipedia", listing("wikimedia/wikipedia", "20231101.ta/")), "text")
    out["sangraha"] = (hub("ai4bharat/sangraha", listing("ai4bharat/sangraha", "verified/tam/")[:sangraha_files]), "text")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_dir", default="data/prepared", help="the valid and test shards here give the references to exclude")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min_count", type=int, default=2, help="minimum word count for unigrams.txt")
    ap.add_argument("--sangraha_files", type=int, default=2)
    ap.add_argument("--max_docs", type=int, default=None, help="documents per source (smoke runs)")
    ap.add_argument("--threads", type=int, default=4, help="parallel file reads")
    ap.add_argument("--discount_fallback", action="store_true", help="pass --discount_fallback to lmplz (tiny corpora)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # The references of valid and test. A missing shard dir must stop the run, so that test text cannot leak into the LM.
    exclude = set()
    for split in ("valid", "test"):
        for files in list_shards(args.exclude_dir, split).values():
            for f in files:
                exclude.update(pq.read_table(f, columns=["text"]).column(0).to_pylist())
    if not exclude:
        sys.exit(f"error: no valid or test shards in {args.exclude_dir}")
    grams = set().union(*map(ngrams, exclude))
    print(f"{len(exclude)} reference texts and {len(grams)} reference 5-grams to exclude", flush=True)

    fs = HfFileSystem()
    text_path = os.path.join(args.out_dir, "text.txt")
    counts = collections.Counter()  # ponytail: the word counts live in memory. Fine up to some 10M distinct words. Upgrade: count with sort | uniq -c.
    seen = set()  # ponytail: 8-byte hashes of the sentences, in memory. A collision drops a sentence at about 1 in 10^12 per pair. Upgrade: sort -u.
    stats = {}
    with open(text_path, "w", encoding="utf-8") as out, ThreadPoolExecutor(args.threads) as ex:
        for name, (files, col) in doc_sources(args.sangraha_files).items():
            st = dict(docs=0, sentences=0, rejected=0, excluded=0, duplicates=0, kept=0, words=0)
            # Files are read in parallel. ex.map keeps the order. Each file is a list of texts in memory.
            for docs in ex.map(lambda p: read_text(fs, p, col, args.max_docs), files):
                for doc in docs[: args.max_docs - st["docs"] if args.max_docs else None]:
                    st["docs"] += 1
                    kept, rej, exc = clean_doc(doc or "", exclude, grams)
                    st["sentences"] += len(kept) + rej + exc
                    st["rejected"] += rej
                    st["excluded"] += exc
                    for s in kept:
                        h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
                        if h in seen:
                            st["duplicates"] += 1
                            continue
                        seen.add(h)
                        out.write(s + "\n")
                        words = s.split()
                        counts.update(words)
                        st["kept"] += 1
                        st["words"] += len(words)
            st["reject_share"] = round(st["rejected"] / max(st["sentences"], 1), 4)
            st["seconds"] = round(time.time() - t0, 1)
            stats[name] = st
            print(name, json.dumps(st), flush=True)

    vocab = [w for w, c in counts.most_common() if c >= args.min_count]  # most frequent first, so head -n N gives a smaller list
    with open(os.path.join(args.out_dir, "unigrams.txt"), "w", encoding="utf-8") as f:
        f.write("".join(w + "\n" for w in vocab))
    if not stats or not any(s["kept"] for s in stats.values()):
        sys.exit("error: no sentence is left for the LM")

    arpa, binary = os.path.join(args.out_dir, "lm.arpa"), os.path.join(args.out_dir, "lm.bin")
    t1 = time.time()
    cmd = [os.path.join(KENLM, "lmplz"), "-o", str(args.order), "--prune", *["0"] + ["1"] * (args.order - 1),
           "-S", "1500M", "-T", args.out_dir, "--text", text_path, "--arpa", arpa]
    if args.discount_fallback:
        cmd.append("--discount_fallback")
    subprocess.run(cmd, check=True)
    subprocess.run([os.path.join(KENLM, "build_binary"), "-q", "8", "-b", "8", "trie", arpa, binary], check=True)
    os.remove(arpa)

    total = {k: sum(s[k] for s in stats.values()) for k in ("docs", "sentences", "rejected", "excluded", "duplicates", "kept", "words")}
    result = dict(sources=stats, total=total, unigrams=len(vocab), distinct_words=len(counts), lm_bin_bytes=os.path.getsize(binary),
                  order=args.order, min_count=args.min_count, lmplz_seconds=round(time.time() - t1, 1), seconds=round(time.time() - t0, 1))
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"unigrams: {len(vocab)}  lm.bin: {result['lm_bin_bytes'] / 1e6:.1f} MB")
    print("@@" + json.dumps(result))


if __name__ == "__main__":
    main()
