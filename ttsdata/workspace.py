"""Filesystem conventions for per-book, per-stage work directories.

Centralising path layout here keeps stages consistent and resumable: a stage is
"done" when its output manifest exists, and ``--force`` simply ignores that.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config

# Ordered stage directory names (number prefix keeps them sorted on disk).
STAGE_DIRS = {
    "ingest": "00_ingest",
    "text_prep": "01_text",
    "transcribe": "02_asr",
    "align": "03_align",
    "segment": "04_segment",
    "quality": "05_quality",
}

# Manifest filename written by each stage inside its directory.
STAGE_MANIFEST = {
    "ingest": "chapters.jsonl",
    "text_prep": "sentences.jsonl",
    "transcribe": "chapters.jsonl",   # chapters enriched with word timestamps
    "align": "segments.jsonl",
    "segment": "clips.jsonl",
    "quality": "clips.jsonl",
}


def raw_book_dir(cfg: Config, book_id: str) -> Path:
    return cfg.path("paths.raw") / book_id


def work_book_dir(cfg: Config, book_id: str) -> Path:
    return cfg.path("paths.work") / book_id


def stage_dir(cfg: Config, book_id: str, stage: str) -> Path:
    return work_book_dir(cfg, book_id) / STAGE_DIRS[stage]


def stage_manifest(cfg: Config, book_id: str, stage: str) -> Path:
    return stage_dir(cfg, book_id, stage) / STAGE_MANIFEST[stage]


def list_books(cfg: Config) -> list[str]:
    """Discover book ids under the raw input directory."""
    raw = cfg.path("paths.raw")
    if not raw.exists():
        return []
    return sorted(p.name for p in raw.iterdir() if p.is_dir())


def discover_chapter_audio(cfg: Config, book_id: str) -> list[Path]:
    """Return chapter audio files for a book, sorted by filename."""
    audio_dir = raw_book_dir(cfg, book_id) / "audio"
    if not audio_dir.exists():
        return []
    exts = {".opus", ".mp3", ".m4a", ".flac", ".wav"}
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)
