"""Stage 0 — ingest & audio normalisation.

Decodes each chapter audio file into two WAV copies:
  * an ASR/align copy (16 kHz mono) for Whisper and forced aligners, and
  * a master copy (22.05 kHz mono 16-bit) that final clips are cut from.

Writes one chapter record per file to the ingest manifest. A book maps to a
single narrator, so every chapter shares one ``speaker_id`` (the book id).
"""

from __future__ import annotations

import logging

from .. import audio
from ..config import Config
from ..manifest import write_jsonl
from ..workspace import (
    discover_chapter_audio,
    stage_dir,
    stage_manifest,
)

log = logging.getLogger(__name__)


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

    records: list[dict] = []
    for order, src in enumerate(audio_files):
        chapter_id = src.stem
        info = audio.probe(src)
        asr_wav = asr_dir / f"{chapter_id}.wav"
        master_wav = master_dir / f"{chapter_id}.wav"

        audio.decode_to_wav(src, asr_wav, asr_sr, channels, sample_format)
        audio.decode_to_wav(src, master_wav, master_sr, channels, sample_format)

        records.append(
            {
                "book_id": book_id,
                "speaker_id": book_id,
                "chapter_id": chapter_id,
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
        )
        log.info("ingest: %s/%s (%.1fs)", book_id, chapter_id, info["duration"])

    write_jsonl(out_path, records)
    total = sum(r["duration"] for r in records)
    log.info("ingest: %s — %d chapters, %.1f min", book_id, len(records), total / 60)
    return records
