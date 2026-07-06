"""Stage 4 — segmentation / cutting.

Cuts the high-quality master WAV at each aligned segment, snapping boundaries to
nearby silence so words aren't clipped, padding slightly, and merging too-short
neighbours. Over-long clips are left for the quality stage to reject (splitting a
single long sentence would require re-aligning its text, out of scope here).

Chapters are cut in parallel (``segment.jobs`` config, 0 = CPU count); the work is
GIL-releasing libsndfile I/O plus numpy, so a thread pool is all we need.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .. import vad
from ..config import Config
from ..manifest import read_jsonl, write_jsonl
from ..workspace import stage_dir, stage_manifest

log = logging.getLogger(__name__)


def _merge_short(segs: list[dict], min_dur: float, max_dur: float, max_gap: float) -> list[dict]:
    """Greedily merge consecutive short segments (concatenating their labels)."""
    merged: list[dict] = []
    cur = None
    for s in segs:
        if cur is None:
            cur = dict(s)
            continue
        cur_dur = cur["end"] - cur["start"]
        gap = s["start"] - cur["end"]
        combined = s["end"] - cur["start"]
        if cur_dur < min_dur and gap <= max_gap and combined <= max_dur:
            cur["text"] = f"{cur['text']} {s['text']}".strip()
            cur["normalized"] = f"{cur['normalized']} {s['normalized']}".strip()
            cur["end"] = s["end"]
            cur["score"] = round((cur["score"] + s["score"]) / 2, 3)
        else:
            merged.append(cur)
            cur = dict(s)
    if cur is not None:
        merged.append(cur)
    return merged


def _process_chapter(
    book_id: str,
    chapter: dict,
    chap_segs: list[dict],
    min_dur: float,
    max_dur: float,
    pad: float,
    wav_dir: Path,
) -> list[dict]:
    """Cut one chapter's master WAV into clips, returning their manifest records."""
    chapter_id = chapter["chapter_id"]
    audio, sr = sf.read(chapter["master_wav"], dtype="float32", always_2d=False)
    audio = vad.to_mono(np.asarray(audio))
    total_s = len(audio) / sr
    # One RMS pass per chapter; boundary snapping then works on frame indices.
    rms, frame_len = vad.frame_rms(audio, sr)
    thr = vad.silence_threshold(rms)

    chap_segs.sort(key=lambda s: s["start"])
    merged = _merge_short(chap_segs, min_dur, max_dur, max_gap=1.0)

    clips: list[dict] = []
    for idx, s in enumerate(merged):
        start = vad.snap_start(rms, frame_len, sr, thr, s["start"])
        end = vad.snap_end(rms, frame_len, sr, thr, s["end"])
        start = max(0.0, start - pad)
        end = min(total_s, end + pad)
        if end <= start:
            continue
        clip = audio[int(start * sr) : int(end * sr)]
        clip_id = f"{chapter_id}_{idx:04d}"
        wav_path = wav_dir / f"{clip_id}.wav"
        sf.write(wav_path, clip, sr, subtype="PCM_16")

        clips.append(
            {
                "book_id": book_id,
                "speaker_id": chapter.get("speaker_id", book_id),
                "chapter_id": chapter_id,
                "seg_id": s["seg_id"],
                "clip_id": clip_id,
                "wav": str(wav_path),
                "text": s["text"],
                "normalized": s["normalized"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "align_score": s["score"],
            }
        )
    return clips


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "segment")
    if out_path.exists() and not force:
        log.info("segment: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    segments = read_jsonl(stage_manifest(cfg, book_id, "align"))
    chapters = {c["chapter_id"]: c for c in read_jsonl(stage_manifest(cfg, book_id, "ingest"))}
    if not segments:
        raise FileNotFoundError(f"No align manifest for {book_id}; run align first.")

    min_dur = float(cfg.get("segment.min_duration", 1.0))
    max_dur = float(cfg.get("segment.max_duration", 15.0))
    pad = float(cfg.get("segment.silence_pad", 0.1))
    jobs = int(cfg.get("segment.jobs", 0)) or os.cpu_count() or 1
    wav_dir = stage_dir(cfg, book_id, "segment") / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Group by chapter so we load each master WAV only once.
    by_chapter: dict[str, list[dict]] = {}
    for s in segments:
        by_chapter.setdefault(s["chapter_id"], []).append(s)

    # Cut chapters in parallel: each worker holds one master in memory and does
    # GIL-releasing libsndfile I/O + numpy, so a thread pool overlaps the waits.
    by_chapter_clips: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for chapter_id, chap_segs in by_chapter.items():
            chapter = chapters.get(chapter_id)
            if chapter is None:
                log.warning("segment: chapter %s missing from ingest; skipping", chapter_id)
                continue
            futures[pool.submit(
                _process_chapter, book_id, chapter, chap_segs, min_dur, max_dur, pad, wav_dir
            )] = chapter_id
        progress = tqdm(
            as_completed(futures), total=len(futures),
            desc=f"segment {book_id}", unit="ch",
        )
        for future in progress:
            by_chapter_clips[futures[future]] = future.result()

    # Restore deterministic order: chapters by ingest order, clips by index within.
    clips: list[dict] = []
    for chapter_id in sorted(by_chapter_clips, key=lambda cid: chapters[cid]["order"]):
        clips.extend(by_chapter_clips[chapter_id])

    write_jsonl(out_path, clips)
    total = sum(c["duration"] for c in clips)
    log.info("segment: %s — %d clips, %.1f min", book_id, len(clips), total / 60)
    return clips
