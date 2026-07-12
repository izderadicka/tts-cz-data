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

First the one **system** dependency (not pip-installable), needed by the ingest
stage to decode opus/mp3:

```bash
apt-get install ffmpeg        # or: brew install ffmpeg
```

Then the Python package. The core install is light (numpy, soundfile, pyyaml,
num2words) and runs text-prep, segment, quality and export. The heavy/situational
pieces are extras, lazily imported so a stage only needs its extra when you run it:

```bash
python -m venv .venv && source .venv/bin/activate   # recommended
python -m pip install -e .            # core (text-prep, segment, quality, export)
python -m pip install -e '.[asr]'     # + faster-whisper for transcription
python -m pip install -e '.[gpu]'     # + torch/torchaudio (CUDA, forced aligner)
python -m pip install -e '.[quality]' # + jiwer for the optional re-ASR CER check
python -m pip install -e '.[all]'     # everything (full pipeline)
```

> Use a virtualenv. Installing into a distro's **system** Python can fail building
> `num2words`'s `docopt` dependency on Debian/Ubuntu (patched setuptools); a venv
> with current `setuptools` avoids this.

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

### Review workflow

Some defects (e.g. a grunt/creak fused to a clip edge — the `edge_noise` flag)
can't be judged automatically; flagged clips get status `review` and are
excluded from export until a human listens:

```bash
ttsdata listen data/work/mybook/05_quality/clips.jsonl --status review --flag edge_noise
# g approves, x rejects; verdicts land in data/work/mybook/review/verdicts.csv
ttsdata review-apply mybook                       # fold verdicts into the manifest
ttsdata run --book mybook --stage export --force  # re-export with them applied
```

Verdicts are re-applied automatically at the end of every quality run, so they
survive `--force` re-runs. Set `quality.reasr_reuse: true` to re-run quality
without re-transcribing every clip (valid while segment output is unchanged).

## Stopgap: Edge TTS proxy

A small, **temporary** utility (`ttsdata/serve/edge_proxy.py`) that exposes
Microsoft Edge's Czech "Read Aloud" neural voices (Antonín, Vlasta) over local
HTTP, so a browser reader (foliate-js) has a decent-sounding Czech voice **while
the Piper Czech voice is being trained**. It is deliberately not part of the
pipeline and lives in its own `serve` extra. A server-side proxy is needed
because browser JavaScript cannot set the WebSocket handshake headers Edge
expects — the browser POSTs text and gets MP3 back.

> ⚠️ Stopgap only. It relies on an **unofficial Microsoft endpoint** (via the
> `edge-tts` library) that may change or break without notice. Delete the
> `serve/` package and the `serve` extra once the Piper voice is ready.

```bash
uv sync --extra serve            # installs fastapi, uvicorn, edge-tts (not core deps)
uv run ttsdata-edge-proxy        # -> http://127.0.0.1:8899
# or: uv run python -m ttsdata.serve.edge_proxy
# host/port: --host/--port flags, or $EDGE_PROXY_HOST / $EDGE_PROXY_PORT
```

Endpoints: `GET /voices` (Czech voices only) and `POST /tts`. Synthesize and
save an MP3:

```bash
curl -s -X POST http://127.0.0.1:8899/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Příliš žluťoučký kůň úpěl ďábelské ódy.", "voice": "cs-CZ-AntoninNeural"}' \
  -o out.mp3
```

`rate` and `pitch` are optional Edge prosody forms (`"+10%"`, `"-5%"`, `"+0Hz"`).
CORS is wide open for local dev; restrict `allow_origins` to the reader's real
origin in production (see the comment in `edge_proxy.py`).

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
