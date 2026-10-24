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
- Decoding: greedy CTC, no language model.

## Results

Word error rate (WER) and character error rate (CER) on the test sets (lower is better for both):

| Test set | WER (lower is better) | CER (lower is better) |
| --- | --- | --- |
| FLEURS ta_in (official, all 591 utts) | 38.4% | 15.1% |
| IISc-MILE (official test, 10,844 utts) | 24.2% | 4.5% |

The references keep digits and Latin letters. The model writes Tamil script only.
So those words always count as errors. This is the main reason why the FLEURS CER is higher.

Ablations on the valid sets at step 1,250 (same data and budget; average WER, lower is better):

- BPE units (1,024) with a frozen encoder: 56.5%. Gemma-token units (3,793) with a frozen encoder: 63.0%.
- BPE units with the top 2 encoder layers fine-tuned: 50.0%.

## How to use

Clone the code repository, [0xAnto/eg2-tamil](https://github.com/0xAnto/eg2-tamil). Install the requirements:

    pip install -r requirements.txt

Run the script. It prints one line for each file: the path, a TAB, and the text.

    python scripts/transcribe.py --ckpt best.pt a.wav b.mp3

The checkpoint holds the reader, the head, the fine-tuned encoder layers, and the units.
It does not hold the other encoder weights. The script downloads them from
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) on the first run (about 1.5 GB).
The script splits audio that is longer than 30 s at the quietest point between 20 s and 30 s, and joins the texts.

## Limitations

- The model was trained and tested on read speech only. Conversational speech and noisy speech are not tested.
- The output has no digits and no Latin script.
- Decoding is greedy. There is no language model.
- Common Voice reuses sentences across speakers. 78% of the test sentences of the Prajwal Common Voice part
  also occur in the train set. The WER on that test set (22.4%) is optimistic. Use the FLEURS and IISc-MILE numbers.

## Credits

- The approach comes from [qnguyen3/gemma-asr](https://github.com/qnguyen3/gemma-asr). This model is a new implementation.
- Google made EmbeddingGemma 2.
- The authors of Prajwal-143/ASR-Tamil-cleaned, IISc-MILE (deepdml/iisc-mile-tamil-asr), and FLEURS made the datasets.

## License

Apache-2.0, the same as the base model. The datasets have their own licenses.
