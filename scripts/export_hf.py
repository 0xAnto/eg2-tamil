#!/usr/bin/env python
"""Export a checkpoint to a release folder for the Hugging Face Hub. This script does not upload anything.

  python scripts/export_hf.py --ckpt runs/main/best.pt --out_dir runs/hf_release [--lm data/lm]

The folder holds model.safetensors (the trainable weights), config.json (config, units, step),
README.md (MODEL_CARD.md), LICENSE and, with --lm, lm/lm.bin and lm/unigrams.txt.
The frozen encoder is not in the folder. load_checkpoint() and GemmaCTC.from_checkpoint() read it from the Hub.
Pass the folder to --ckpt of transcribe.py or decode_lm.py.
"""

import argparse
import json
import os
import shutil
import sys

from safetensors.torch import save_file

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from eg2tamil.model import load_checkpoint  # noqa: E402


def export(ckpt: dict, out_dir: str) -> None:
    """Write model.safetensors and config.json for the checkpoint dict `ckpt` into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    # safetensors refuses tensors that share memory, so each tensor gets its own contiguous copy.
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in ckpt["model"].items()},
              os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": ckpt["config"], "units": ckpt["units"], "step": ckpt["step"]}, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a .pt file from train.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lm", default=None, help="directory with lm.bin and unigrams.txt. The script copies both files")
    args = ap.parse_args()

    export(load_checkpoint(args.ckpt), args.out_dir)
    shutil.copyfile(os.path.join(ROOT, "MODEL_CARD.md"), os.path.join(args.out_dir, "README.md"))
    shutil.copyfile(os.path.join(ROOT, "LICENSE"), os.path.join(args.out_dir, "LICENSE"))
    if args.lm:
        os.makedirs(os.path.join(args.out_dir, "lm"), exist_ok=True)
        for name in ("lm.bin", "unigrams.txt"):
            shutil.copyfile(os.path.join(args.lm, name), os.path.join(args.out_dir, "lm", name))

    for dirpath, _, names in sorted(os.walk(args.out_dir)):
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            print(f"{os.path.getsize(path):>14,d}  {os.path.relpath(path, args.out_dir)}")


if __name__ == "__main__":
    main()
