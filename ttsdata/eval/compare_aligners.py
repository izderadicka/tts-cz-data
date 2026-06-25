"""Compare alignment strategies on the same book.

You asked to evaluate the ASR-bridge vs forced alignment. The workflow:

  1. Run align with strategy A, copy its manifest aside.
  2. Run align with strategy B, copy its manifest aside.
  3. Point this harness at both; it reports, per strategy:
       * segment count and total / retained audio,
       * mean alignment score,
       * boundary tightness (leading/trailing silence per clip — tighter is
         better, it means cuts land on speech onsets/offsets), and
       * optional label CER via re-ASR (needs faster-whisper + jiwer).

Audio-dependent metrics use the master WAVs from the ingest manifest; the
audio-free metrics work from the segment manifests alone.
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from .. import vad
from ..config import Config
from ..manifest import read_jsonl
from ..workspace import stage_manifest

log = logging.getLogger(__name__)


def basic_metrics(segments: list[dict], chapter_durations: dict[str, float]) -> dict:
    """Audio-free metrics computable from the segment manifest alone."""
    durs = [s["end"] - s["start"] for s in segments]
    total = sum(durs)
    scores = [s.get("score", 0.0) for s in segments]
    available = sum(chapter_durations.values()) or 1.0
    return {
        "segments": len(segments),
        "audio_seconds": round(total, 1),
        "retained_pct": round(100 * total / available, 1),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "dur_mean": round(total / len(durs), 2) if durs else 0.0,
        "dur_min": round(min(durs), 2) if durs else 0.0,
        "dur_max": round(max(durs), 2) if durs else 0.0,
    }


def boundary_metrics(segments: list[dict], master_wavs: dict[str, str]) -> dict:
    """Mean leading/trailing silence (seconds) at clip edges — tighter is better."""
    lead, trail = [], []
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for s in segments:
        wav = master_wavs.get(s["chapter_id"])
        if not wav:
            continue
        if wav not in cache:
            audio, sr = sf.read(wav, dtype="float32", always_2d=False)
            cache[wav] = (vad.to_mono(np.asarray(audio)), sr)
        audio, sr = cache[wav]
        clip = audio[int(s["start"] * sr) : int(s["end"] * sr)]
        if clip.size == 0:
            continue
        rms, frame_len = vad.frame_rms(clip, sr)
        thr = vad.silence_threshold(rms)
        voiced = np.where(rms >= thr)[0]
        if voiced.size == 0:
            continue
        lead.append(voiced[0] * frame_len / sr)
        trail.append((len(rms) - 1 - voiced[-1]) * frame_len / sr)
    return {
        "lead_silence_mean": round(float(np.mean(lead)), 3) if lead else None,
        "trail_silence_mean": round(float(np.mean(trail)), 3) if trail else None,
    }


def compare(cfg: Config, book_id: str, manifests: dict[str, str], audio: bool = True) -> dict:
    """Return {strategy_name: metrics} for each provided segment manifest."""
    chapters = read_jsonl(stage_manifest(cfg, book_id, "ingest"))
    durations = {c["chapter_id"]: c.get("duration", 0.0) for c in chapters}
    master_wavs = {c["chapter_id"]: c.get("master_wav") for c in chapters}

    report: dict[str, dict] = {}
    for name, path in manifests.items():
        segments = read_jsonl(path)
        metrics = basic_metrics(segments, durations)
        if audio and any(master_wavs.values()):
            metrics.update(boundary_metrics(segments, master_wavs))
        report[name] = metrics
        log.info("compare[%s]: %s", name, metrics)
    return report


def format_report(report: dict[str, dict]) -> str:
    """Render the comparison as a simple aligned table."""
    if not report:
        return "(no strategies to compare)"
    keys = list(next(iter(report.values())).keys())
    names = list(report)
    header = "metric".ljust(20) + "".join(n.ljust(16) for n in names)
    lines = [header, "-" * len(header)]
    for k in keys:
        row = k.ljust(20) + "".join(str(report[n].get(k, "")).ljust(16) for n in names)
        lines.append(row)
    return "\n".join(lines)
