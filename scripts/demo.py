#!/usr/bin/env python
"""A local web page to try the model. Record or upload Tamil speech, then read the text without and with the LM.

  python scripts/demo.py --ckpt runs/hf_release --lm data/lm

The page needs gradio (pip install gradio). gradio is not in requirements.txt.
"""

import argparse
import os
import sys

import gradio as gr
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decode_lm import build_decoder  # noqa: E402
from eg2tamil.model import GemmaCTC, greedy_decode, load_checkpoint  # noqa: E402
from eg2tamil.text import normalize_hypothesis  # noqa: E402
from eg2tamil.units import Units  # noqa: E402
from transcribe import load, split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a .pt file or a release folder (see export_hf.py)")
    ap.add_argument("--lm", default=None, help="directory with lm.bin and unigrams.txt")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    model, units = GemmaCTC.from_checkpoint(ckpt).eval(), Units(ckpt["units"])
    decoder = build_decoder(units, args.lm) if args.lm else None

    @torch.no_grad()
    def run(path):
        if not path:
            return "", ""
        greedy, beam = [], []
        for c in split(load(path)):  # ponytail: one chunk at a time on the CPU. A demo clip is short.
            logits, n = model(torch.from_numpy(c)[None], torch.tensor([len(c)]))
            greedy.append(normalize_hypothesis(units.decode(greedy_decode(logits, n)[0])))
            if decoder:
                lp = torch.log_softmax(logits.float(), -1)[0, : int(n[0])].numpy()
                beam.append(normalize_hypothesis(decoder.decode(lp, beam_width=64)))
        return " ".join(greedy), " ".join(beam) if decoder else "(start the demo with --lm)"

    # Blocks, not Interface: Interface puts the two texts one above the other. Side by side, the reader compares them line by line.
    with gr.Blocks(title="eg2-tamil") as page:
        gr.Markdown("# eg2-tamil\nTamil speech recognition on the EmbeddingGemma 2 audio encoder. Record or upload a clip.")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tamil speech")
        submit = gr.Button("Submit", variant="primary")
        with gr.Row():
            texts = [gr.Textbox(label="Greedy (the model only)"), gr.Textbox(label="With LM")]
        submit.click(run, audio, texts)
    page.launch(server_port=args.port)


if __name__ == "__main__":
    main()
