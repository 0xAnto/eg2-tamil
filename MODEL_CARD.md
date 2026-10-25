---
language:
- ta
license: apache-2.0
library_name: pytorch
pipeline_tag: automatic-speech-recognition
base_model: google/embeddinggemma-2
datasets:
- Prajwal-143/ASR-Tamil-cleaned
- deepdml/iisc-mile-tamil-asr
- google/fleurs
- ai4bharat/sangraha
- wikimedia/wikipedia
tags:
- ctc
- tamil
- speech-recognition
- embeddinggemma
metrics:
- wer
- cer
model-index:
- name: eg2-tamil
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: FLEURS ta_in
      type: google/fleurs
      config: ta_in
      split: test
    metrics:
    - type: wer
      value: 38.4
      name: WER (lower is better)
    - type: cer
      value: 15.1
      name: CER (lower is better)
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: IISc-MILE Tamil ASR
      type: deepdml/iisc-mile-tamil-asr
      split: test
    metrics:
    - type: wer
      value: 24.2
      name: WER (lower is better)
    - type: cer
      value: 4.5
      name: CER (lower is better)
---

# eg2-tamil

eg2-tamil is a Tamil speech recognition model. It reads 16 kHz speech and writes Tamil text.
It is a small CTC "reader" on top of the audio encoder of
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

## Architecture

- The audio encoder of EmbeddingGemma 2 has 12 conformer layers and makes 25 frames per second.
  The bottom 10 layers are frozen. The top 2 layers are fine-tuned.
- A learned mix combines the 13 activations of the encoder.
- The reader is a 6-layer transformer with d=512.
- The CTC head has one row for each of 1,024 Tamil BPE units.
  Each row starts from the Gemma text embeddings of its unit.
- The model has 70.6M trainable parameters.

## Training data

462 h of Tamil speech:

| Source | Hours |
| --- | --- |
| Prajwal-143/ASR-Tamil-cleaned, Common Voice part | 340 |
| Prajwal-143/ASR-Tamil-cleaned, MILE part | 116 |
| google/fleurs (ta_in) | 6 |

The data preparation drops every clip id that it has already seen.
It reads the test ids first, then the valid ids, then the train ids.
So no test clip or valid clip is in the train set, and no clip occurs twice.
The rule works on ids, not on text. See "Limitations".

## Training

- One NVIDIA L4, bf16.
- 18,270 steps of 600 s of padded audio. This is about 6.3 epochs.
- AdamW, warmup, then cosine decay to the time budget.
- SpecAugment.
- Decoding: greedy CTC. A word n-gram LM is optional (see "Results").

## Results

Word error rate (WER) and character error rate (CER) on the test sets (lower is better for both):

| Test set | WER (lower is better) | CER (lower is better) |
| --- | --- | --- |
| FLEURS ta_in (official, all 591 utts) | 38.4% | 15.1% |
| IISc-MILE (official test, 10,844 utts) | 24.2% | 4.5% |

The references keep digits and Latin letters. The model writes Tamil script only.
So those words always count as errors. This is the main reason why the FLEURS CER is higher.

Ablations on the valid sets at step 1,250 (same data and warmup; average WER, lower is better). The two frozen-encoder
runs had a 35-min budget. The fine-tuned value comes from the 6-h main run, so its LR decays more slowly: at step 1,250
it was 0.00098, not 0.00071. That comparison is not controlled:

- BPE units (1,024) with a frozen encoder: 56.5%. Gemma-token units (3,793) with a frozen encoder: 63.0%.
- BPE units with the top 2 encoder layers fine-tuned: 50.0%.

With beam search and a KenLM 4-gram word LM (pyctcdecode; `--alpha 0.3 --beta 0.5 --unk_offset -5`, tuned on the
valid sets). The LM is in `lm/` of this repository. Lower is better for both:

| Test set | WER greedy | WER with LM | CER greedy | CER with LM |
| --- | --- | --- | --- | --- |
| FLEURS ta_in | 38.5% | 31.6% | 15.2% | 13.8% |
| IISc-MILE | 24.3% | 21.7% | 4.5% | 3.8% |

The LM is a word 4-gram (KenLM, 366 MB). Its text is 74.5M words: the train transcripts, Tamil Wikipedia, and 2 files
of AI4Bharat Sangraha. The LM text does not contain a valid or test sentence, or a sequence of 5 words from one.
The decoder needs about 0.7 GB of RAM for the LM.
The LM evaluation keeps test clips of up to 120 s (IISc-MILE: 10,845 utts), so its greedy numbers are 0.1 higher.

The model-index above gives the greedy numbers, because those need no LM.

## How to use

The code is in [0xAnto/eg2-tamil](https://github.com/0xAnto/eg2-tamil) on GitHub. Clone it, install the requirements,
and download this model repository:

    git clone https://github.com/0xAnto/eg2-tamil && cd eg2-tamil
    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/hf download 0xanto/eg2-tamil --local-dir eg2-tamil-model
    .venv/bin/python scripts/transcribe.py --ckpt eg2-tamil-model audio.wav

The script prints one line for each file: the path, a TAB, and the text. It cuts audio that is longer than 30 s at
quiet points, and then joins the texts. A CPU is sufficient: on an 8-core laptop CPU, the model runs at 11–16x
realtime (higher is better).

For better results, use the LM in `lm/`. Install the LM packages, then add `--lm`:

    uv pip install --python .venv/bin/python --no-deps pyctcdecode pygtrie https://github.com/kpu/kenlm/archive/master.zip
    .venv/bin/python scripts/transcribe.py --ckpt eg2-tamil-model --lm eg2-tamil-model/lm audio.wav

`--no-deps` is necessary: pyctcdecode asks for numpy < 2, but it works with numpy 2.

To try the model on a local web page (microphone or file upload), see the "Demo" section of the
[README](https://github.com/0xAnto/eg2-tamil#demo) on GitHub.

This repository holds the reader, the head, the fine-tuned encoder layers, and the units (`model.safetensors`,
`config.json`). It does not hold the frozen encoder layers. The script downloads them from
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) on the first run (about 1.5 GB).

## Limitations

- The model was trained and tested on read speech only. Conversational speech and noisy speech are not tested.
- The output has no digits and no Latin script.
- Common Voice reuses sentences across speakers. 78% of the test sentences of the Prajwal Common Voice part
  also occur in the train set. The WER on that test set (22.4%) is optimistic. Use the FLEURS and IISc-MILE numbers.

## Credits

- The idea comes from [qnguyen3/gemma-asr](https://github.com/qnguyen3/gemma-asr) (Vietnamese speech recognition
  with EmbeddingGemma 2). Thank you to its author.
- Google made EmbeddingGemma 2.
- The speech data comes from Mozilla Common Voice (CC0), the MILE lab at IISc (CC BY 2.0), Google FLEURS (CC BY 4.0),
  and the `Prajwal-143/ASR-Tamil-cleaned` dataset.
- The LM text comes from Tamil Wikipedia (CC BY-SA) and AI4Bharat Sangraha (CC BY 4.0).

## License

- The model files (`model.safetensors`, `config.json`) use Apache-2.0, the same as the base model.
- The LM files (`lm/lm.bin`, `lm/unigrams.txt`) use CC BY-SA 4.0, because the LM text includes Tamil Wikipedia.
  The model files do not need the LM.
