# eg2-tamil

An open-source experiment: Tamil speech recognition (speech to text) on top of the audio encoder of
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

**The question.** EmbeddingGemma 2 has an audio encoder that Google made for embeddings, not for speech recognition.
Can a small model on top of this encoder learn to write Tamil? How good is it with public data and one GPU?

**The short answer.** Yes. On the public FLEURS Tamil benchmark, it is better than Whisper large-v3, but it is not
yet as good as the best dedicated Indian-language models.

> Status: research experiment, not a product. The model weights and the LM are on Hugging Face:
> [0xanto/eg2-tamil](https://huggingface.co/0xanto/eg2-tamil).

## Results

The model listens to 16 kHz speech and writes Tamil script. We test it on two public test sets. The model never heard
their recordings in training:

- **FLEURS Tamil** (`google/fleurs`, `ta_in`, test split): 591 recordings (2.1 h) of read sentences from
  Wikipedia-style articles. This is the standard public benchmark, so other papers report numbers on it.
- **IISc-MILE Tamil** (`deepdml/iisc-mile-tamil-asr`, test split): 10,845 recordings (15.1 h) of read sentences,
  recorded by the MILE lab at the Indian Institute of Science (IISc).

| Test set | Word error rate, greedy | Word error rate, with LM | Character error rate, with LM |
| --- | --- | --- | --- |
| FLEURS Tamil | 38.5% | **31.6%** | 13.8% |
| IISc-MILE Tamil | 24.3% | **21.7%** | 3.8% |

Lower is better for all columns.

- **Word error rate (WER):** the number of wrong, missing and extra words for every 100 words of the reference.
  A WER of 31.6% means that roughly 7 of 10 words come out exactly right. Tamil words are long, and one wrong letter
  makes the full word wrong. So the **character error rate (CER)** is also important: on IISc-MILE, about 96 of
  100 letters are correct.
- **Greedy** is the model alone. **With LM** adds a beam search with a Tamil word 4-gram language model (LM). The LM runs on
  a normal CPU and needs no GPU.

FLEURS Tamil, compared with other systems (WER, lower is better). The other numbers come from the
[osmapi/tamil-asr-qwen3](https://huggingface.co/osmapi/tamil-asr-qwen3) model card. That card uses its own text
normalizer, so the comparison is approximate.

| System | FLEURS Tamil WER |
| --- | --- |
| Qwen3-ASR, fine-tuned for Tamil | 25.3% |
| IndicConformer-600M (AI4Bharat) | 25.5% |
| **eg2-tamil, with LM** | **31.6%** |
| eg2-tamil, greedy | 38.5% |
| Whisper large-v3 | 48.6% |

### Examples

Five FLEURS Tamil test clips of 4–10 s were selected at random (seed 0), not selected for good results. These are
the first three. Each example gives the FLEURS file id, so you can listen to the clip.

**1.** `14512923187739791106` (8.0 s). Meaning: "Usually you can always hear the noise of tourists and vendors here.
The story of the sound and light show is like a storybook."

| | Text | WER |
| --- | --- | --- |
| Reference | வழக்கமாக நீங்கள் எப்போதும் இங்கே சுற்றுலா பயணிகள் மற்றும் விற்பனையாளர்களின் ஒலியை கேட்கலாம் ஒலி மற்றும் ஒளியின் கதையானது ஒரு கதை புத்தகம் போன்றது | |
| Greedy | வழக்கமாக நீங்கள் எப்போதும் எங்கே சுற்றுலாப் பயணிகள் மற்றும் விற்பனையாளர்களின் ஒியை கேட்கலாம் ஒளி மற்றும் மொழின் கதையானது ஒரு கதை புத்தகம் போன்றது | 28% |
| With LM | வழக்கமாக நீங்கள் எப்போதும் இங்கே சுற்றுலாப் பயணிகள் மற்றும் விற்பனையாளர்களின் ஒ கேட்கலாம் ஒளி மற்றும் மொழியின் கதையானது ஒரு கதை புத்தகம் போன்றது | 22% |

**2.** `13871963685686168470` (8.4 s). Meaning: "The Sundarbans forests are a UNESCO World Heritage Site. The part of the
Sundarbans inside the Indian border is called the Sundarbans National Park."

| | Text | WER |
| --- | --- | --- |
| Reference | சுந்தரவனக் காடுகள் யுனெஸ்கோவின் உலக பாரம்பரிய தளமாக அறிவிக்கப்பட்டுள்ளது இந்திய எல்லைக்குள் உள்ள சுந்தரவனத்தின் பகுதி சுந்தர்வன தேசிய பூங்கா என்று அழைக்கப்படுகிறது | |
| Greedy | சுந்தருக்குடுகள் யூனோஸோவின் உலக பாரம்பரி தளமாக அறிவிக்கப்பட்டுள்ளது இந்திய எல்லைக்குள் உள்ள சுந்த வத்தின் பகுதி சுந்தரண தேசியி பூ என்று அழைக்கப்படுகிறது | 53% |
| With LM | சுந்தர ஆடுகள் யுனெஸ்கோவின் உலக பாரம்பரிய தளமாக அறிவிக்கப்பட்டுள்ளது இந்திய எல்லைக்குள் உள்ள சுந்தரத்தின் பகுதி சுந்தர தேசிய பூ என்று அழைக்கப்படுகிறது | 29% |

**3.** `17264457532158271660` (5.3 s). Meaning: "The scenes are shown on the pyramids, and different pyramids are lit up."

| | Text | WER |
| --- | --- | --- |
| Reference | காட்சிகள் பிரமிடுகளில் காட்சிப்படுத்தப்பட்டு வெவ்வேறு பிரமிடுகள் ஒளியேற்றப்படும் | |
| Greedy | காட்சிகள் பெருமிடுகளில் காட்சிப்படுத்தப்பட்டு வெவ்வேறு பெருமிடுகள் வெளியேற்றப்படும் | 50% |
| With LM | காட்சிகள் பெருமிடுகளில் காட்சிப்படுத்தப்பட்டு வெவ்வேறு பிரமிடுகள் வெளியேற்றப்படும் | 33% |

The examples show the typical errors. The model hears the sounds well, but it misspells names and rare words
(`யூனோஸோவின்` for `யுனெஸ்கோவின்`, "UNESCO's"). It also drops letters at the end of a word. The LM corrects many of
these errors. It cannot correct a word that sounds similar and is also a real word (`வெளியேற்றப்படும்`, "will be
removed", for `ஒளியேற்றப்படும்`, "will be lit up").

## What we learned

1. **The embedding encoder works for speech recognition.** The model learns Tamil with only a small reader and a
   CTC head on top of the encoder. Fine-tuning the top 2 encoder layers probably helps too (that comparison is not
   controlled; see the ablations).
2. **Small Tamil subword units are better than Gemma's own text tokens.** 1,024 BPE units gave 56.5% valid WER,
   and the 3,793 Gemma tokens gave 63.0% at the same training budget (lower is better).
3. **A word n-gram LM helps a lot,** and it costs no GPU: −6.9 WER points on FLEURS and −2.6 on IISc-MILE.
4. **For Tamil, the most important LM setting is the penalty for unknown words.** Tamil has very many word forms,
   so 4–6% of the correct words are not in the LM vocabulary. The default penalty (−10) is too strong. −5 is better.
5. **More LM text helps only a little.** 3.2 times more web text (74M to 236M words) improved the test WER by only
   0.1 (FLEURS) and 0.2 (IISc-MILE) points. The larger LM also needs 2.6 times more disk space and 2.4 times more RAM.
6. **Data leaks are easy to miss.** Common Voice reuses sentences across speakers: 78% of its test sentences
   are also in its train set. Web text also contains test sentences: before we added a filter, every 5-word sequence
   of 1.3% of the IISc-MILE test sentences was in the LM text.

## How it works

```
16 kHz audio
  -> log-mel features (128 bins)
  -> EmbeddingGemma 2 audio encoder: 12 conformer layers, 25 frames per second (top 2 layers fine-tuned)
  -> a learned mix of the 13 encoder outputs
  -> reader: 6 transformer layers, d = 512
  -> CTC head: 1,024 Tamil BPE units (each row starts from the Gemma text embeddings of its unit)
  -> greedy decoding, or beam search with a KenLM word 4-gram LM
  -> Tamil text
```

The model has 70.6M trainable parameters: the reader, the head, and the top 2 encoder layers.

## Try it

You need Python 3.12. A CPU is sufficient for transcription: on an 8-core laptop CPU, the model runs at 11–16x
realtime (higher is better). The commands below download the model and the LM from Hugging Face (about 0.7 GB). The
first run also downloads the encoder (about 1.5 GB).

    git clone https://github.com/0xAnto/eg2-tamil && cd eg2-tamil
    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/hf download 0xanto/eg2-tamil --local-dir eg2-tamil-model
    .venv/bin/python scripts/transcribe.py --ckpt eg2-tamil-model audio.wav

The script prints one line for each file: the path, a TAB, and the text. It reads the formats of libsndfile (wav,
flac; mp3 with libsndfile 1.1 or later). It cuts audio that is longer than 30 s at quiet points, and then joins the
texts.

For better results, use the LM in `eg2-tamil-model/lm`. Install the LM packages, then add `--lm`:

    uv pip install --python .venv/bin/python --no-deps pyctcdecode pygtrie https://github.com/kpu/kenlm/archive/master.zip
    .venv/bin/python scripts/transcribe.py --ckpt eg2-tamil-model --lm eg2-tamil-model/lm audio.wav

`--no-deps` is necessary: pyctcdecode asks for numpy < 2, but it works with numpy 2.

## Demo

`scripts/demo.py` is a small web page to try the model. You record Tamil speech with the microphone or upload an
audio file. The page then shows two texts: the model only (greedy), and the model with the LM.

First do the steps in [Try it](#try-it). Then install gradio and start the page:

    .venv/bin/pip install gradio
    .venv/bin/python scripts/demo.py --ckpt eg2-tamil-model --lm eg2-tamil-model/lm

Open http://127.0.0.1:7860 in your browser. The page runs on your computer only. Without `--lm`, it shows the
greedy text only.

## Reproduce

### Data

462 h of public Tamil speech:

| Source | Speech | Hours |
| --- | --- | --- |
| Common Voice Tamil, from `Prajwal-143/ASR-Tamil-cleaned` | read sentences, crowd-sourced volunteers | 340 |
| IISc-MILE Tamil, from `Prajwal-143/ASR-Tamil-cleaned` | read sentences | 116 |
| FLEURS Tamil (`google/fleurs`, `ta_in`), train split | read sentences | 6 |

    python scripts/prepare.py --sources prajwal mile fleurs --raw_dir data/raw --out_dir data/prepared

- The script downloads only the files that it needs and writes clean 16 kHz FLAC shards. It takes about 25 min on a
  12-CPU machine.
- It reads the test clip ids first, then the valid ids, then the train ids, and it drops every id that it has
  already seen. So no test clip is in the train set, and no clip occurs twice.
- Train text must be Tamil script only. Test text keeps digits and Latin letters, so that the test sets stay complete
  and comparable with published numbers. The model cannot write digits, so these words always count as errors.
- `--splits valid test` prepares only the evaluation data (about 5.5 GB of downloads).

Each dataset has its own license. FLEURS is CC-BY-4.0. See the dataset pages for the others.

### Train and evaluate

Training needs one GPU with bf16 support (L4, A100). A T4 does not work. The main run used one NVIDIA L4: 18,270
steps in a 355-min training budget, and about 8.9 h of GPU time in total with the data preparation and the
evaluation.

    python scripts/train.py --data_dir data/prepared --units data/prepared/units_bpe.json --out_dir runs/main \
        --reader base --unfreeze_top_k 2 --warmup_steps 500 --max_minutes 355
    python scripts/train.py --data_dir data/prepared --eval_only --resume runs/main/best.pt --eval_split test

- The learning rate schedule follows the time budget (`--max_minutes`). The run evaluates, saves and stops when the
  time is up.
- The run stops with an error if no optimizer step ends for 10 minutes, or if the loss is not finite for 50 steps
  in a row.
- The checkpoint holds the model config and the units, so the evaluation does not need the units file.
- `--resume runs/main/last.pt` continues a run (`last.pt` holds the optimizer state).
- The last line of every run starts with `@@` and holds one JSON object with the final metrics.

Other flags: `--grad_accum N`, `--train_sources`, `--source_weights '{"mile": 0.5}'`, `--no_specaug`,
`--head_init random`, `--no_layer_mix`, `--encoder_layers K` (run only the first K encoder layers).

Reader sizes (`--reader`, presets in `READERS` in `eg2tamil/model.py`; `--d_model`, `--n_layers` and `--n_heads`
override a preset):

| Preset | d_model | Layers | Heads |
| --- | --- | --- | --- |
| small | 256 | 4 | 4 |
| base | 512 | 6 | 8 |
| large | 768 | 8 | 12 |

To add a dataset, add one entry to `SOURCES` in `eg2tamil/sources.py`: the Hugging Face repo, the parquet glob of
each split, the text column and the id column. Then run `scripts/prepare.py --sources <name>`. Each source gets its
own valid and test set in the evaluation.

### Build the LM

The LM needs the KenLM tools. Clone [kpu/kenlm](https://github.com/kpu/kenlm) to `tools/kenlm`, then build `lmplz`
and `build_binary` with `cmake -DCMAKE_BUILD_TYPE=Release`. With Homebrew boost 1.90, first remove `system` from
`find_package(Boost ...)` in `tools/kenlm/CMakeLists.txt`.

    python scripts/build_lm.py --out_dir data/lm

- The LM text is 8.0M sentences (74.5M words): the train transcripts, Tamil Wikipedia (`wikimedia/wikipedia`,
  `20231101.ta`) and the first 2 of 53 files of `ai4bharat/sangraha` (verified Tamil). The build takes about 25 min,
  mostly for the downloads.
- The text goes through the same rule as the train text, so 23–31% of the web sentences are rejected (digits, Latin
  letters). Duplicate sentences are removed.
- **No test text in the LM.** The script needs `data/prepared/valid` and `data/prepared/test`. It removes every
  sentence that equals a valid or test sentence, or that shares a sequence of 5 words with one.
- The LM is a 4-gram (singletons pruned for orders 2–4, 8-bit quantized trie): `lm.bin` is 366 MB. `unigrams.txt`
  has the 1.48M words that occur 2 or more times. The decoder needs about 0.7 GB of RAM for it. This is the LM on
  Hugging Face.

### Tune and score the LM

    python scripts/decode_lm.py --ckpt runs/main/best.pt --data_dir data/prepared --split valid --lm data/lm \
        --alpha 0.2 0.3 0.4 --beta 0 0.5 1 --unk_offset -3 -5 -10

- The first run computes the model outputs one time and keeps them in `runs/logits/`. Each later setting reads them
  from there, so one setting takes 20–30 s for 4,377 valid clips.
- The settings were tuned on the valid sets only: `--alpha 0.3 --beta 0.5 --unk_offset -5 --beam 64`. These are the
  defaults of `decode_lm.py` and `transcribe.py`. Then the test sets were decoded one time.
- A wider beam does not help: beam 64, 128 and 256 give the same valid WER within 0.03 points.

## Detailed results

All test sets, greedy decoding (lower is better for WER and CER):

| Test set | Clips | Hours | Test text also in train | WER | CER |
| --- | --- | --- | --- | --- | --- |
| FLEURS Tamil | 591 | 2.1 | 0% | 38.4% | 15.1% |
| IISc-MILE Tamil | 10,844 | 15.1 | 0.9% | 24.2% | 4.5% |
| Common Voice Tamil (Prajwal test split) | 22,269 | 38.5 | 78% | 22.4% | 4.2% |
| IISc-MILE Tamil (Prajwal test split) | 8,920 | 14.9 | 1.3% | 25.5% | 4.9% |

- FLEURS Tamil and IISc-MILE Tamil are the clean numbers. The Common Voice number is too good, because 78% of its
  test sentences are also in the train set.
- The LM evaluation uses `prepare.py --splits valid test`, which keeps test clips of up to 120 s. So IISc-MILE has 1
  clip more there (10,845), and its greedy WER is 24.3%.

The two LM sizes, with the same decoder settings (lower is better for `lm.bin`, decoder RAM and WER):

| LM | LM text | `lm.bin` | Decoder RAM | FLEURS WER | IISc-MILE WER |
| --- | --- | --- | --- | --- | --- |
| None (greedy) | | | | 38.5% | 24.3% |
| 2 Sangraha files (on Hugging Face) | 74.5M words | 366 MB | 0.7 GB | 31.6% | 21.7% |
| 8 Sangraha files | 236M words | 942 MB | 1.7 GB | 31.5% | 21.5% |

A word list with all the words of the LM text (4.46M words, also the words that occur only one time) helps only a
little. With the small LM, the mean valid WER of FLEURS Tamil and IISc-MILE goes from 25.3% to 25.0% (lower is
better; best setting `--alpha 0.4`). But the decoder then needs about 1.4 GB more RAM. So the LM keeps the words
that occur 2 or more times.

The LM tuning target is the mean WER of two valid sets with little train overlap (FLEURS Tamil and IISc-MILE from
Prajwal). Common Voice valid is not in the target, because 77% of its sentences are also in the train text. With the
default unknown-word penalty (−10), the LM made Common Voice worse (21.4% to 22.6%). With −5, it makes Common Voice
better (20.7%).

The LM filter for test text is important for honest numbers. Before the 5-word filter, every 5-word sequence of 1.3%
of the IISc-MILE test sentences was in the LM text. That leaky LM (2 Sangraha files) gave IISc-MILE 21.6% instead of
21.7%, so the effect was small. After the filter, no valid or test sentence shares a sequence of 5 words with the
LM text.

Ablations on the valid sets at step 1,250 (average WER, lower is better):

| Change | Average valid WER |
| --- | --- |
| Gemma-token units (3,793), frozen encoder | 63.0% |
| BPE units (1,024), frozen encoder | 56.5% |
| BPE units, top 2 encoder layers fine-tuned | 50.0% |

The two frozen-encoder runs had a 35-min budget. The fine-tuned value comes from the 6-h main run, so its learning
rate decays more slowly (0.00098 at step 1,250, not 0.00071). Thus the last row is not a controlled comparison.

Other findings:

- The learned layer mix puts most weight on the top encoder layers (0.34 on the last one at the end). So
  `--encoder_layers` below 12 is not useful for this encoder.
- `torch.compile` does not work here: it compiles again for each input length (see the note in `scripts/train.py`).
- Training speed on an L4 (higher is better): 550x realtime with a frozen encoder, 435x realtime with the top 2
  layers fine-tuned.

## Limitations

- The model was trained and tested on read speech only. Conversational speech and noisy speech are not tested.
- The model writes Tamil script only: no digits, no Latin letters, no punctuation.
- The results come from one training run. We did not measure the spread across random seeds.
- The LM settings were tuned on read speech. Other speech can need other settings.

## Open questions

These are good next experiments. Pull requests are welcome.

- **More encoder layers:** fine-tune the top 4 layers instead of 2, in a controlled comparison.
- **Numbers:** convert digits in the train text to Tamil words, so that the rejected rows can be used.
- **More data:** for example, other public Tamil speech datasets.
- **Unknown words:** more web text helped only a little. A subword or character LM can help more with the many
  Tamil word forms.

## Contributing

1. For a large change, open an issue first, and describe the change.
2. Run the tests. They need no GPU. `tests/test_model.py` downloads the encoder on the first run.

       for t in tests/test_*.py; do .venv/bin/python $t || break; done

3. Keep each pull request to one topic, and keep the code minimal.
4. If a pull request changes a result, give the command, the data, and a table. Each table must say which direction
   is better ("lower is better" for WER). Tune only on the valid sets, and report how much test text is also in
   the train data or the LM text.

## Project layout

    eg2tamil/   text.py, sources.py, units.py, data.py, scoring.py (data side), model.py (model side)
    scripts/    prepare.py (data), train.py (train and evaluate), transcribe.py (inference),
                build_lm.py (the n-gram LM), decode_lm.py (beam search with the LM),
                demo.py (a local web page to try the model), export_hf.py (the Hugging Face release folder)
    tests/      test_data.py, test_model.py, test_transcribe.py, test_lm.py, test_build_lm.py, test_export.py
    colab/      job.sh, stage.sh (run the stages on a Colab VM)

`MODEL_CARD.md` is the model card for the Hugging Face release.

## Credits

- The idea comes from [qnguyen3/gemma-asr](https://github.com/qnguyen3/gemma-asr) (Vietnamese speech recognition
  with EmbeddingGemma 2). Thank you to its author.
- Google made [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).
- The speech data comes from Mozilla Common Voice, the MILE lab at IISc, Google FLEURS, and the
  `Prajwal-143/ASR-Tamil-cleaned` dataset.
- The LM text comes from Tamil Wikipedia and AI4Bharat Sangraha.
- The decoding uses [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) and
  [KenLM](https://github.com/kpu/kenlm).

## License

The code and the model weights use the Apache-2.0 license (see `LICENSE`). The base model `google/embeddinggemma-2`
also uses Apache-2.0. Each dataset has its own license. The LM text includes Tamil Wikipedia (CC BY-SA), so the LM
files on Hugging Face (`lm/`) use CC BY-SA 4.0.
