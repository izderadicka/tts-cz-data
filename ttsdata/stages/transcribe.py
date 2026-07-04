"""Stage 2 — transcription (faster-whisper) with word-level timestamps.

Each chapter's 16 kHz ASR WAV is transcribed in Czech. We keep the noisy ASR
text and, crucially, **word timestamps** — these are the bridge that lets the
ASR-bridge aligner anchor clean book sentences to positions on the audio clock.

The model/device/compute type are config-driven so the same code runs on a CPU
(``small``/``int8``) or scales to CUDA (``large-v3``/``float16``).
"""

from __future__ import annotations

import logging

from ..config import Config
from ..manifest import read_jsonl, write_jsonl
from ..workspace import stage_manifest

log = logging.getLogger(__name__)

_model_cache: dict = {}


def _resolve_device(cfg: Config) -> tuple[str, str]:
    device = cfg.get("asr.device", "cpu")
    compute_type = cfg.get("asr.compute_type", "int8")
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"
    return device, compute_type


def _get_model(cfg: Config):
    from faster_whisper import WhisperModel

    model_name = cfg.get("asr.model", "small")
    device, compute_type = _resolve_device(cfg)
    key = (model_name, device, compute_type)
    if key not in _model_cache:
        log.info("loading faster-whisper '%s' on %s (%s)", model_name, device, compute_type)
        _model_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model_cache[key]


def transcribe_file(cfg: Config, wav_path: str) -> dict:
    """Transcribe one WAV; return {text, words:[{start,end,word,prob}]}."""
    model = _get_model(cfg)
    segments, _info = model.transcribe(
        wav_path,
        language=cfg.get("asr.language", "cs"),
        beam_size=int(cfg.get("asr.beam_size", 5)),
        vad_filter=bool(cfg.get("asr.vad_filter", True)),
        word_timestamps=bool(cfg.get("asr.word_timestamps", True)),
    )
    words: list[dict] = []
    texts: list[str] = []
    for seg in segments:
        texts.append(seg.text)
        for w in (seg.words or []):
            words.append(
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                    "prob": round(getattr(w, "probability", 0.0) or 0.0, 3),
                }
            )
    return {"text": "".join(texts).strip(), "words": words}


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "transcribe")
    if out_path.exists() and not force:
        log.info("transcribe: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    chapters = read_jsonl(stage_manifest(cfg, book_id, "ingest"))
    if not chapters:
        raise FileNotFoundError(f"No ingest manifest for {book_id}; run ingest first.")

    for ch in chapters:
        result = transcribe_file(cfg, ch["asr_wav"])
        ch["asr_text"] = result["text"]
        ch["words"] = result["words"]
        log.info("transcribe: %s/%s — %d words", book_id, ch["chapter_id"], len(result["words"]))

    write_jsonl(out_path, chapters)
    return chapters
