"""Stage 2 — transcription (faster-whisper) with word-level timestamps.

Each chapter's 16 kHz ASR WAV is transcribed in Czech. We keep the noisy ASR
text and, crucially, **word timestamps** — these are the bridge that lets the
ASR-bridge aligner anchor clean book sentences to positions on the audio clock.

The model/device/compute type are config-driven so the same code runs on a CPU
(``small``/``int8``) or scales to CUDA (``large-v3``/``float16``).

On CPU, chapters are transcribed in parallel: ``asr.jobs`` worker threads share
one model created with ``num_workers=jobs`` (CTranslate2 runs the calls truly
concurrently), each using ``asr.cpu_threads`` intra-op threads.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

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


def _resolve_jobs(cfg: Config, device: str) -> int:
    """Number of chapters to transcribe concurrently (0 in config = auto)."""
    jobs = int(cfg.get("asr.jobs", 0))
    if jobs:
        return jobs
    if device != "cpu":
        return 1  # parallel GPU workers would need extra VRAM; keep sequential
    cpu_threads = int(cfg.get("asr.cpu_threads", 4)) or 4
    return max(1, (os.cpu_count() or 1) // cpu_threads)


def _get_model(cfg: Config):
    from faster_whisper import WhisperModel

    model_name = cfg.get("asr.model", "small")
    device, compute_type = _resolve_device(cfg)
    cpu_threads = int(cfg.get("asr.cpu_threads", 4))
    num_workers = _resolve_jobs(cfg, device)
    key = (model_name, device, compute_type, cpu_threads, num_workers)
    if key not in _model_cache:
        log.info(
            "loading faster-whisper '%s' on %s (%s, %d workers x %d threads)",
            model_name, device, compute_type, num_workers, cpu_threads,
        )
        _model_cache[key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
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


def _transcribe_chapter(cfg: Config, ch: dict) -> dict:
    """Worker task: transcribe one chapter and mutate its record in place."""
    result = transcribe_file(cfg, ch["asr_wav"])
    ch["asr_text"] = result["text"]
    ch["words"] = result["words"]
    return ch


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "transcribe")
    if out_path.exists() and not force:
        log.info("transcribe: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    chapters = read_jsonl(stage_manifest(cfg, book_id, "ingest"))
    if not chapters:
        raise FileNotFoundError(f"No ingest manifest for {book_id}; run ingest first.")

    # Load the model up front so worker threads share one instance instead of
    # racing the cache.
    _get_model(cfg)
    jobs = _resolve_jobs(cfg, _resolve_device(cfg)[0])

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_transcribe_chapter, cfg, ch) for ch in chapters]
        progress = tqdm(
            as_completed(futures), total=len(futures),
            desc=f"transcribe {book_id}", unit="ch",
        )
        for future in progress:
            ch = future.result()
            log.debug(
                "transcribe: %s/%s — %d words", book_id, ch["chapter_id"], len(ch["words"])
            )

    # Workers mutate the ordered `chapters` list in place, so output order
    # matches the serial run regardless of completion order.
    write_jsonl(out_path, chapters)
    total_words = sum(len(c["words"]) for c in chapters)
    log.info(
        "transcribe: %s — %d chapters, %d words", book_id, len(chapters), total_words
    )
    return chapters
