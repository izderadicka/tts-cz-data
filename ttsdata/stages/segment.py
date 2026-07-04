"""Stage 4 — segmentation / cutting.

Cuts the high-quality master WAV at each aligned segment, snapping boundaries to
nearby silence so words aren't clipped, padding slightly, and merging too-short
neighbours. Over-long clips are left for the quality stage to reject (splitting a
single long sentence would require re-aligning its text, out of scope here).
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

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
    wav_dir = stage_dir(cfg, book_id, "segment") / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Group by chapter so we load each master WAV only once.
    by_chapter: dict[str, list[dict]] = {}
    for s in segments:
        by_chapter.setdefault(s["chapter_id"], []).append(s)

    clips: list[dict] = []
    for chapter_id, chap_segs in by_chapter.items():
        chapter = chapters.get(chapter_id)
        if chapter is None:
            log.warning("segment: chapter %s missing from ingest; skipping", chapter_id)
            continue
        audio, sr = sf.read(chapter["master_wav"], dtype="float32", always_2d=False)
        audio = vad.to_mono(np.asarray(audio))
        total_s = len(audio) / sr

        chap_segs.sort(key=lambda s: s["start"])
        merged = _merge_short(chap_segs, min_dur, max_dur, max_gap=1.0)

        for idx, s in enumerate(merged):
            start = vad.snap_to_silence(audio, sr, s["start"], window_s=0.2)
            end = vad.snap_to_silence(audio, sr, s["end"], window_s=0.2)
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

    write_jsonl(out_path, clips)
    total = sum(c["duration"] for c in clips)
    log.info("segment: %s — %d clips, %.1f min", book_id, len(clips), total / 60)
    return clips
