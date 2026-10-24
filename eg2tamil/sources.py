"""Dataset registry. To add a dataset, add one entry to SOURCES."""

import fnmatch
import functools
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from huggingface_hub.utils import build_hf_headers, get_session


def stem(x: str) -> str:
    """Basename of an id or path, without the extension."""
    return os.path.splitext(os.path.basename(x))[0]


# repo: HF dataset id. splits: our split name -> parquet glob inside the repo.
# revision: optional Hub revision. text/id: column names. out: optional function that maps an id to the output source name (default: the source key).
SOURCES = {
    "prajwal": dict(
        repo="Prajwal-143/ASR-Tamil-cleaned",
        splits={"train": "data/train-*.parquet", "valid": "data/validation-*.parquet", "test": "data/test-*.parquet"},
        text="sentence",
        id="path",
        out=lambda i: "prajwal_cv" if i.startswith("common_voice") else "prajwal_mile",
    ),
    "mile": dict(
        repo="deepdml/iisc-mile-tamil-asr",
        splits={"train": "data/train-*.parquet", "test": "data/test-*.parquet"},
        text="transcription",
        id="id",
    ),
    # FLEURS Tamil test is the public benchmark. Its id is the file name: the "id" column is the sentence id,
    # and several speakers read the same sentence. The text is raw_transcription (the same result as transcription after normalize).
    "fleurs": dict(
        repo="google/fleurs",
        revision="refs/convert/parquet",
        splits={"train": "ta_in/train/*.parquet", "valid": "ta_in/validation/*.parquet", "test": "ta_in/test/*.parquet"},
        text="raw_transcription",
        id="path",
    ),
}


def out_name(name: str, id_: str) -> str:
    return SOURCES[name].get("out", lambda _: name)(id_)


@functools.cache
def _repo_files(repo: str, revision: str | None) -> list[str]:
    return sorted(HfApi().list_repo_files(repo, repo_type="dataset", revision=revision))


def remote_files(name: str, split: str, max_files: int | None = None) -> list[str]:
    """Parquet paths inside the repo for one split, sorted."""
    return fnmatch.filter(_repo_files(SOURCES[name]["repo"], SOURCES[name].get("revision")), SOURCES[name]["splits"][split])[:max_files]


def read_ids(name: str, rel: str, raw_dir: str, max_rows: int | None = None) -> tuple[list[str], int]:
    """Read only the id column of one parquet file. Return (ids, file size in bytes).
    Use the local copy if it exists, else read from the Hub."""
    spec, local = SOURCES[name], os.path.join(raw_dir, name, rel)
    if os.path.exists(local):
        t, size = pq.read_table(local, columns=[spec["id"]]), os.path.getsize(local)
    else:
        # Fetch the footer and the id column chunks with parallel range requests.
        # They go into a sparse in-memory copy of the file. The rest of the copy stays empty and pyarrow never reads it.
        url, sess = hf_hub_url(spec["repo"], rel, repo_type="dataset", revision=spec.get("revision")), get_session()

        def get(rng):
            for wait in (1, 2, 4, 8, 16, 32, 64, 0):  # the Hub answers 429 when requests come too fast
                r = sess.get(url, headers={**build_hf_headers(), "Range": f"bytes={rng}"})
                if r.status_code != 429 or not wait:
                    break
                time.sleep(float(r.headers.get("Retry-After", wait)))
            r.raise_for_status()
            return r

        r = get(f"-{1 << 16}")
        size, tail = int(r.headers["Content-Range"].split("/")[1]), r.content
        foot = int.from_bytes(tail[-8:-4], "little") + 8
        if foot > len(tail):
            tail = get(f"-{foot}").content
        buf = mmap.mmap(-1, size)
        buf[size - len(tail) :] = tail
        md = pq.read_metadata(pa.BufferReader(pa.py_buffer(buf)))
        c = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)].index(spec["id"])
        ranges = []
        for g in range(md.num_row_groups):
            m = md.row_group(g).column(c)
            start = m.dictionary_page_offset if m.has_dictionary_page else m.data_page_offset
            ranges.append((start, start + m.total_compressed_size))

        def fetch(r):
            buf[r[0] : r[1]] = get(f"{r[0]}-{r[1] - 1}").content

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(fetch, ranges))
        t = pq.ParquetFile(pa.BufferReader(pa.py_buffer(buf))).read(columns=[spec["id"]])
    return [stem(i) for i in t.column(0).to_pylist()[:max_rows]], size


def download(name: str, raw_dir: str, rels: list[str]) -> None:
    """Download parquet files into raw_dir/<name>. Files that exist are skipped, partial files resume."""
    if rels:
        snapshot_download(SOURCES[name]["repo"], repo_type="dataset", allow_patterns=rels,
                          revision=SOURCES[name].get("revision"), local_dir=os.path.join(raw_dir, name))
