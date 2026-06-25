# tts-cz-data

A staged pipeline that turns **Czech audiobook chapters** (opus/mp3) plus the
**corresponding book** (plain text) into a **TTS training dataset**: short audio
clips (1–15 s) each paired with an accurate transcript, exported in the universal
**LJSpeech** format that Piper, Coqui/VITS, Tacotron, etc. consume directly.

The pipeline is **config-driven**, **manifest-based** (every stage reads/writes
JSONL so it is inspectable), and **resumable** (a stage is skipped if its output
exists, unless `--force`). It runs on **CPU by default** and scales to **CUDA**
by changing config only.

## Pipeline stages

| # | Stage | What it does |
|---|-------|--------------|
| 0 | `ingest` | ffmpeg-decode chapters to 16 kHz (ASR) + 22.05 kHz (master) WAVs |
| 1 | `text_prep` | clean book text, Czech sentence split, normalise to spoken form |
| 2 | `transcribe` | faster-whisper ASR with word timestamps (language `cs`) |
| 3 | `align` | map book sentences to audio time spans (2 strategies, pluggable) |
| 4 | `segment` | cut clips at silence-snapped boundaries, enforce length bounds |
| 5 | `quality` | re-ASR + audio checks; pass / flag-for-review / reject |
| 6 | `export` | LJSpeech `metadata.csv` + wavs, train/val split, dataset stats |

### Alignment strategies (compared via `ttsdata/eval/compare_aligners.py`)
- **`asr_bridge`** *(default)*: sequence-align Whisper words against the clean book
  text; the book sentence is the ground-truth label. Robust to narrator edits.
- **`forced`**: direct forced alignment (aeneas / torchaudio-MMS / MFA) for precise
  word boundaries.

## Input layout

```
data/raw/<book_id>/
  book.txt              # full book, plain UTF-8 text
  audio/                # chapter files, sorted by filename
    01.opus
    02.opus
    ...
```

## Install

```bash
python -m pip install -e .            # core skeleton (fast)
python -m pip install -e '.[asr,align,text,vad,quality,dev]'   # full pipeline
```

System dependency: **ffmpeg** (`apt-get install ffmpeg` / `brew install ffmpeg`).

## Run

```bash
ttsdata list-books
ttsdata run --stage ingest --book mybook     # one stage, one book
ttsdata run --book mybook                     # all stages for a book
ttsdata run                                   # all books, all stages
ttsdata run --config config/cuda.yaml --book mybook   # GPU override
```

All knobs live in [`config/default.yaml`](config/default.yaml); pass `--config`
with a small override file to change models, device, thresholds, etc.

## Status

Implemented and tested end to end (synthetic audio):
- infrastructure: config loader, JSONL manifests, CLI orchestrator, workspace layout
- **stage 0 ingest** (ffmpeg) · **stage 1 text-prep** (Czech split + normalisation)
- **stage 2 transcribe** (faster-whisper) · **stage 3 align — ASR-bridge strategy**
- **stage 4 segment** (silence-snapped cutting) · **stage 5 quality** (+ flagged.csv)
- **stage 6 export** (LJSpeech + train/val + stats + Piper notes)
- **eval/compare_aligners** harness for comparing strategies

To complete: **stage 3 forced-alignment strategy** (torchaudio-MMS / aeneas / MFA)
— a guided stub in `ttsdata/stages/align/forced.py` (heavy deps + real audio
needed to validate, so not shipped untested).

Run the tests with `python -m pytest`.
