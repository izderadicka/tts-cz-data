"""Stage 0 — ingest & audio normalisation.

Decodes each chapter audio file into two WAV copies:
  * an ASR/align copy (16 kHz mono) for Whisper and forced aligners, and
  * a master copy (22.05 kHz mono 16-bit) that final clips are cut from.

Writes one chapter record per file to the ingest manifest. A book maps to a
single narrator, so every chapter shares one ``speaker_id`` (the book id).

Chapters are processed in parallel (``ingest.jobs`` config, 0 = CPU count);
the work is ffmpeg subprocesses, so a thread pool is all we need.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .. import audio
from ..config import Config
from ..manifest import write_jsonl
from ..workspace import (
    discover_chapter_audio,
    stage_dir,
    stage_manifest,
)

log = logging.getLogger(__name__)


def _process_chapter(
    book_id: str,
    order: int,
    src: Path,
    asr_wav: Path,
    master_wav: Path,
    asr_sr: int,
    master_sr: int,
    channels: int,
    sample_format: str,
) -> dict:
    """Probe one chapter and decode its two WAV copies (in parallel)."""
    info = audio.probe(src)
    with ThreadPoolExecutor(max_workers=2) as pool:
        asr_fut = pool.submit(
            audio.decode_to_wav, src, asr_wav, asr_sr, channels, sample_format
        )
        master_fut = pool.submit(
            audio.decode_to_wav, src, master_wav, master_sr, channels, sample_format
        )
        asr_fut.result()
        master_fut.result()

    return {
        "book_id": book_id,
        "speaker_id": book_id,
        "chapter_id": src.stem,
        "order": order,
        "src_path": str(src),
        "asr_wav": str(asr_wav),
        "master_wav": str(master_wav),
        "duration": info["duration"],
        "codec": info["codec"],
        "src_sample_rate": info["sample_rate"],
        "sample_rate_asr": asr_sr,
        "sample_rate_master": master_sr,
    }


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "ingest")
    if out_path.exists() and not force:
        log.info("ingest: %s already done (use --force to redo)", book_id)
        from ..manifest import read_jsonl
        return read_jsonl(out_path)

    audio_files = discover_chapter_audio(cfg, book_id)
    if not audio_files:
        raise FileNotFoundError(
            f"No chapter audio found under {cfg.path('paths.raw')/book_id/'audio'}"
        )

    work = stage_dir(cfg, book_id, "ingest")
    asr_dir = work / "asr"
    master_dir = work / "master"
    asr_sr = int(cfg.require("audio.asr_sample_rate"))
    master_sr = int(cfg.require("audio.master_sample_rate"))
    channels = int(cfg.get("audio.channels", 1))
    sample_format = cfg.get("audio.sample_format", "s16")
    jobs = int(cfg.get("ingest.jobs", 0)) or os.cpu_count() or 1

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                _process_chapter,
                book_id,
                order,
                src,
                asr_dir / f"{src.stem}.wav",
                master_dir / f"{src.stem}.wav",
                asr_sr,
                master_sr,
                channels,
                sample_format,
            )
            for order, src in enumerate(audio_files)
        ]
        progress = tqdm(
            as_completed(futures), total=len(futures),
            desc=f"ingest {book_id}", unit="ch",
        )
        for future in progress:
            rec = future.result()
            records.append(rec)
            log.debug(
                "ingest: %s/%s (%.1fs)", book_id, rec["chapter_id"], rec["duration"]
            )

    records.sort(key=lambda r: r["order"])
    write_jsonl(out_path, records)
    total = sum(r["duration"] for r in records)
    log.info("ingest: %s — %d chapters, %.1f min", book_id, len(records), total / 60)
    return records
