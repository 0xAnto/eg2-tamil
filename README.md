# eg2-tamil

Tamil speech recognition with a small CTC "reader" on top of the frozen audio encoder of
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (`google/embeddinggemma-2`, Apache-2.0).

The approach comes from [qnguyen3/gemma-asr](https://github.com/qnguyen3/gemma-asr) (Vietnamese).
This repository is a new implementation. It does not copy that code.

## How it works

1. A batched log-mel module on the GPU makes 128-bin features. It matches the Hugging Face feature extractor.
2. The frozen audio encoder (12 conformer layers, 25 frames per second) makes 13 activations.
3. A learned softmax mix combines the activations.
4. The reader (a Linear layer, a depthwise convolution, and N pre-norm transformer blocks) maps the mix to text space.
5. The CTC head has one row per unit. Each row starts as the mean Gemma token embedding of its unit.

## Layout

    eg2tamil/   text.py, sources.py, units.py, data.py, scoring.py (data side), model.py (model side)
    scripts/    prepare.py (data), train.py (train and evaluate), transcribe.py (inference)
    tests/      test_model.py, test_data.py, test_transcribe.py

## Setup

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python tests/test_model.py      # downloads the encoder (about 1.5 GB) on the first run

Training needs one GPU with bf16 support (L4, A100). A T4 does not work. The CPU is for smoke tests only.

## Commands

    # 1. prepare the data (writes data/prepared and the units files)
    python scripts/prepare.py --sources prajwal mile fleurs --raw_dir data/raw --out_dir data/prepared

    # 2. train (the default units file is <data_dir>/units_gemma.json)
    python scripts/train.py --data_dir data/prepared --out_dir runs/base --reader base --max_minutes 240

    # 3. evaluate on the test sets (every source in <data_dir>/test)
    python scripts/train.py --data_dir data/prepared --eval_only --resume runs/base/best.pt --eval_split test

    # 4. transcribe audio files (prints path, TAB, text; splits audio longer than 30 s)
    python scripts/transcribe.py --ckpt runs/main/best.pt a.wav b.mp3

The model card for Hugging Face is `MODEL_CARD.md`.

The last line of every run starts with `@@` and holds one JSON object with the final metrics.
The checkpoint holds the model config and the units, so evaluation does not need the units file.
Use `--units data/prepared/units_bpe.json` to train with the BPE units.

Useful flags: `--grad_accum N`, `--max_minutes M` (eval, save and exit when the time is up),
`--resume PATH`, `--train_sources`, `--source_weights '{"mile": 0.5}'`, `--no_specaug`, `--head_init random`,
`--no_layer_mix`, `--encoder_layers K` (run only the first K encoder layers), `--unfreeze_top_k K`.
The run exits with a non-zero code if no optimizer step ends for 10 minutes (`--stall_minutes`),
or if the loss is not finite for 50 steps in a row.

## Add a dataset

Add one entry to `SOURCES` in `eg2tamil/sources.py`: the Hugging Face repo, the parquet glob of each split,
the text column and the id column. Then run `scripts/prepare.py --sources <name>`.
Each source gets its own `valid/<source>` and `test/<source>` set in the evaluation.

## Change the reader size

Use `--reader small|base|large`. The presets are in `READERS` in `eg2tamil/model.py`.

| Preset | d_model | Layers | Heads |
| ------ | ------- | ------ | ----- |
| small  | 256     | 4      | 4     |
| base   | 512     | 6      | 8     |
| large  | 768     | 8      | 12    |

`--d_model`, `--n_layers` and `--n_heads` override the preset. A resumed run reads the size from the checkpoint.

## Results (first run, 2026-10-24)

One Colab L4 for about 8.9 h in total (13.6 CU). Data: 462 h of training audio (Prajwal CV 340 h, Prajwal MILE 116 h,
FLEURS 6 h). Model: `base` reader, BPE units (1,024), top 2 encoder layers fine-tuned (70.6M trainable parameters).
Training: 18,270 steps of 600 s padded audio (about 6.3 epochs), 6.6 h. Decoding: greedy CTC, no language model.

Test sets (lower is better for WER and CER):

| Test set | Utts | Hours | Test text also in train | WER | CER |
| --- | --- | --- | --- | --- | --- |
| FLEURS ta_in (official, all 591 utts) | 591 | 2.1 | 0% | 38.4% | 15.1% |
| IISc-MILE (official test) | 10,844 | 15.1 | 0.9% | 24.2% | 4.5% |
| Prajwal Common Voice | 22,269 | 38.5 | 78% | 22.4% | 4.2% |
| Prajwal MILE | 8,920 | 14.9 | 1.3% | 25.5% | 4.9% |

- FLEURS and IISc-MILE are the clean numbers. Common Voice reuses sentences across speakers, so 78% of the Prajwal CV
  test sentences also occur in train, and its WER is optimistic.
- The eval references keep digits and Latin letters (`normalize(strict=False)`). The model writes Tamil script only,
  so those words always count as errors. This is the main reason why the FLEURS CER is higher than the others.
- Published FLEURS-ta WER for comparison (from the osmapi/tamil-asr-qwen3 model card, their normalizer):
  IndicConformer-600M 25.5%, a Qwen3-ASR fine-tune 25.3%, Whisper large-v3 48.6%.

Ablations on the valid sets at step 1,250 (same data and budget; average WER, lower is better):

| Change | Average valid WER |
| --- | --- |
| Gemma-token units (3,793), frozen encoder | 63.0% |
| BPE units (1,024), frozen encoder | 56.5% |
| BPE units, top 2 encoder layers fine-tuned | 50.0% |

Other findings:
- The learned layer mix puts most weight on the top encoder layers (0.34 on the last one at the end), so
  `--encoder_layers` below 12 is not useful for this encoder.
- `torch.compile` does not work here (see the note in `scripts/train.py`).
- Speed on an L4 (higher is better): 550x realtime with a frozen encoder, 435x realtime with the top 2 layers fine-tuned.

To continue training, resume from `runs/main/last.pt` (it holds the optimizer state).
