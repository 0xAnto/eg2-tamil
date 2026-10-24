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
    scripts/    prepare.py (data), train.py (train and evaluate), transcribe.py (inference), decode_lm.py (beam search with an LM)
    tests/      test_model.py, test_data.py, test_transcribe.py, test_lm.py

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
    python scripts/transcribe.py --ckpt runs/main/best.pt --lm data/lm a.wav   # beam search with the n-gram LM (step 5)

    # 5. build the word n-gram LM (needs data/prepared/{valid,test}: their texts are kept out of the LM).
    #    KenLM tools: clone kpu/kenlm to tools/kenlm, cmake -DCMAKE_BUILD_TYPE=Release, make lmplz build_binary.
    #    With Homebrew boost 1.90, delete "system" from find_package(Boost ...) in tools/kenlm/CMakeLists.txt.
    python scripts/build_lm.py --out_dir data/lm

    # 6. score greedy decoding and beam search with an LM. Install the LM packages first. --no-deps keeps numpy 2
    #    (pyctcdecode pins numpy<2, but it works with numpy 2):
    uv pip install --no-deps pyctcdecode pygtrie https://github.com/kpu/kenlm/archive/master.zip
    python scripts/decode_lm.py --ckpt runs/main/best.pt --data_dir data/prepared --split valid --lm data/lm --alpha 0.2 0.3 0.4 --beta 0 0.5 1 --unk_offset -3 -5 -10

DIR (for example `data/lm`) holds `lm.bin` (the KenLM model) and `unigrams.txt` (one word per line, most frequent first).
`decode_lm.py` caches the log-probabilities in `runs/logits/<ckpt>_<data_dir name>_<split>/`. Without `--lm` it reports the greedy row only.
`prepare.py --splits valid test` prepares the eval splits only.

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

Ablations on the valid sets at step 1,250 (same data and warmup; average WER, lower is better). The two frozen-encoder
runs had a 35-min budget. The fine-tuned value comes from the 6-h main run, so its LR decays more slowly: at step 1,250
it was 0.00098, not 0.00071. That comparison is not controlled:

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

### Beam search with a word n-gram LM (2026-10-24)

The same model (`runs/main/best.pt`), decoded with pyctcdecode and a KenLM 4-gram word LM. No GPU: the log-probabilities
are computed one time on a CPU, then each decoder setting reads them from the cache.

Test sets (lower is better for WER and CER):

| Test set | Utts | WER greedy | WER with LM | CER greedy | CER with LM |
| --- | --- | --- | --- | --- | --- |
| FLEURS ta_in (official, all 591 utts) | 591 | 38.5% | 31.6% | 15.2% | 13.8% |
| IISc-MILE (official test) | 10,845 | 24.3% | 21.7% | 4.5% | 3.8% |

- Settings: `--alpha 0.3 --beta 0.5 --unk_offset -5 --beam 64` (the defaults of `decode_lm.py` and `transcribe.py`).
- These eval sets come from `prepare.py --splits valid test`. It keeps eval clips up to 120 s, so IISc-MILE has
  1 clip more than the first table. The greedy numbers agree with the first table.
- Decode time on a laptop CPU (lower is better): 49 s for the 11,436 test utts with 4 processes. The model pass (about 15x realtime)
  takes most of the time.

LM text (`scripts/build_lm.py`), 7.98M sentences and 74.5M words after the filters:

| Source | Sentences kept | Words |
| --- | --- | --- |
| Train transcripts (Prajwal, FLEURS train) | 0.14M | 1.1M |
| Tamil Wikipedia (20231101.ta, CC-BY-SA) | 1.96M | 16.2M |
| ai4bharat/sangraha verified/tam, first 2 of 53 files (CC-BY-4.0) | 5.88M | 57.2M |

- The text goes through the train rule (`normalize(strict=True)`), so 23–31% of the web sentences are rejected
  (digits, Latin letters). Exact duplicates are removed.
- No eval text in the LM: a sentence is removed when it equals a valid or test reference, or when it shares a word
  5-gram with one. The exact rule alone was not sufficient: for 1.3% of the IISc-MILE test references (5 words or more),
  all word 5-grams were in the LM text. After the 5-gram rule, no eval reference has a 5-gram in the LM text. The leaky LM gave IISc-MILE 21.6%,
  so the leak changed the result by only 0.1 point.
- 4-gram, singletons pruned for orders 2–4, 8-bit quantized trie: `lm.bin` is 366 MB. `unigrams.txt` has the 1.48M words
  that occur 2 or more times. With this list, 5–7% of the reference words are out of vocabulary (OOV).

Tuning, on the valid sets only (105 settings; about 20 s for each):
- The target is the mean WER of FLEURS valid and Prajwal MILE valid: 30.4% greedy, 25.3% at the chosen setting (lower is better).
  Prajwal CV valid is not in the target, because 77% of its sentences are also in the acoustic train text.
  With the LM at the first settings tried (`unk_offset -10`), its WER went up (21.4% to 22.6%). At the chosen
  setting it goes down (20.7%).
- The OOV score has a large effect. At alpha 0.2 and 0.3, -5 is better than the pyctcdecode default (-10) for every beta,
  and the 8 best settings all use -5 or -3. A probable reason: Tamil has many word forms, so a correct word is often
  not in the unigram list.
- A wider beam does not help (lower is better): beam 64, 128 and 256 give the same valid WER within 0.03 points.

To continue training, resume from `runs/main/last.pt` (it holds the optimizer state).
