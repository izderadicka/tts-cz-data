"""Stage 5 — quality filtering with a flagged-for-review export.

Each clip gets cheap audio checks (duration, clipping, silence ratio, rough
SNR, edge noise) and, optionally, an ASR round-trip check: re-transcribe the
clip and compare to its label via character error rate (CER). Outcome is one
of:

  * ``reject`` — a hard failure (out-of-bounds duration, very high CER, ...)
  * ``review`` — a soft flag worth a human look (written to review/flagged.csv)
  * ``pass``   — keep as-is

The ``edge_noise`` check (:func:`ttsdata.vad.edge_noise_ms`) looks for speaker
noise (grunt/creak) fused to a clip edge — silero counts it as speech, so it
survives segmentation. It can't be told apart from legitimate prevoicing
automatically, so it soft-flags for auditioning with ``ttsdata listen``; the
verdicts recorded there (``review/verdicts.csv``) are re-applied at the end of
every run, so human decisions survive ``--force`` re-runs. Use
``ttsdata review-apply`` to fold fresh verdicts in without a re-run.

The re-ASR check needs faster-whisper + jiwer and is off by default
(``quality.reasr_check: false``) so the stage runs cheaply on CPU. With
``quality.reasr_reuse: true`` the cer/asr_text of a previous run are reused
per clip, making a re-run (e.g. after tuning edge thresholds) CPU-only —
valid only while the segment stage output is unchanged.

Clips are checked in parallel (``quality.jobs`` config, 0 = auto). With re-ASR
enabled the worker count follows the transcribe stage's ASR sizing (workers
share one Whisper model; CTranslate2 runs the calls truly concurrently),
otherwise it defaults to the CPU count for the cheap I/O-bound checks.
"""

from __future__ import annotations

import csv
import logging
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .. import review, vad
from ..config import Config
from ..manifest import read_jsonl, write_jsonl
from ..workspace import stage_dir, stage_manifest, work_book_dir

log = logging.getLogger(__name__)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join("".join(ch for ch in folded if ch.isalnum() or ch.isspace()).split())


def _audio_metrics(audio: np.ndarray, sr: int) -> dict:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "peak": round(peak, 4),
        "silence_ratio": round(vad.silence_ratio(audio, sr), 3),
        "snr_db": round(vad.estimate_snr_db(audio, sr), 2),
    }


def _cer(reference: str, hypothesis: str) -> float | None:
    try:
        import jiwer
    except ImportError:
        log.warning("jiwer not installed; skipping CER check")
        return None
    ref, hyp = _fold(reference), _fold(hypothesis)
    if not ref:
        return None
    return float(jiwer.cer(ref, hyp))


def _check_clip(
    cfg: Config,
    clip: dict,
    min_dur: float,
    max_dur: float,
    max_cer: float,
    review_cer: float,
    min_snr: float,
    max_sil: float,
    reasr: bool,
    edge: dict | None,
    prev_asr: dict[str, tuple] | None,
) -> dict:
    """Worker task: run all checks on one clip and mutate its record in place."""
    flags: list[str] = []
    reject = False
    audio, sr = sf.read(clip["wav"], dtype="float32", always_2d=False)
    audio = vad.to_mono(np.asarray(audio))
    m = _audio_metrics(audio, sr)
    clip.update(m)

    if clip["duration"] < min_dur or clip["duration"] > max_dur:
        flags.append("duration"); reject = True
    if m["peak"] >= 0.999:
        flags.append("clipping")
    if m["silence_ratio"] > max_sil:
        flags.append("silence"); reject = True
    if m["snr_db"] < min_snr:
        flags.append("low_snr")

    if edge is not None:
        common = dict(
            frame_ms=edge["frame_ms"], strong_rel=edge["strong_rel"],
            weak_rel=edge["weak_rel"], max_centroid_hz=edge["max_centroid_hz"],
        )
        lead = vad.edge_noise_ms(audio, sr, min_gap_ms=edge["lead_gap_ms"], **common)
        tail = vad.edge_noise_ms(audio[::-1], sr, min_gap_ms=edge["tail_gap_ms"], **common)
        clip["edge_lead_ms"] = round(lead)
        clip["edge_tail_ms"] = round(tail)
        if lead >= edge["min_ms"] or tail >= edge["min_ms"]:
            flags.append("edge_noise")

    if reasr:
        if prev_asr is not None and clip["clip_id"] in prev_asr:
            cer, hyp = prev_asr[clip["clip_id"]]
        else:
            from .transcribe import transcribe_file

            hyp = transcribe_file(cfg, clip["wav"])["text"]
            cer = _cer(clip["normalized"], hyp)
        clip["cer"] = None if cer is None else round(cer, 3)
        clip["asr_text"] = hyp
        if cer is not None:
            if cer > max_cer:
                flags.append("cer_high"); reject = True
            elif cer > review_cer:
                flags.append("cer_review")

    clip["flags"] = flags
    clip["status"] = "reject" if reject else ("review" if flags else "pass")
    return clip


def _resolve_jobs(cfg: Config, reasr: bool) -> int:
    """Worker count (0 in config = auto). With re-ASR the pool must match the
    ASR model's concurrency, so reuse the transcribe stage's sizing."""
    jobs = int(cfg.get("quality.jobs", 0))
    if jobs:
        return jobs
    if reasr:
        from . import transcribe

        return transcribe._resolve_jobs(cfg, transcribe._resolve_device(cfg)[0])
    return os.cpu_count() or 1


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "quality")
    if out_path.exists() and not force:
        log.info("quality: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    clips = read_jsonl(stage_manifest(cfg, book_id, "segment"))
    if not clips:
        raise FileNotFoundError(f"No segment manifest for {book_id}; run segment first.")

    min_dur = float(cfg.get("segment.min_duration", 1.0))
    max_dur = float(cfg.get("segment.max_duration", 15.0))
    max_cer = float(cfg.get("quality.max_cer", 0.20))
    review_cer = float(cfg.get("quality.review_cer", 0.10))
    min_snr = float(cfg.get("quality.min_snr_db", 10.0))
    max_sil = float(cfg.get("quality.max_silence_ratio", 0.5))
    reasr = bool(cfg.get("quality.reasr_check", False))
    edge = None
    if cfg.get("quality.edge_noise_check", True):
        edge = {
            "frame_ms": int(cfg.get("quality.edge_frame_ms", 10)),
            "strong_rel": float(cfg.get("quality.edge_strong_rel", 0.30)),
            "weak_rel": float(cfg.get("quality.edge_weak_rel", 0.05)),
            "max_centroid_hz": float(cfg.get("quality.edge_max_centroid_hz", 900)),
            "min_ms": float(cfg.get("quality.edge_min_ms", 40)),
            "lead_gap_ms": float(cfg.get("quality.edge_lead_gap_ms", 0)),
            "tail_gap_ms": float(cfg.get("quality.edge_tail_gap_ms", 30)),
        }

    # Optionally reuse cer/asr_text from the previous run so re-checking after
    # a threshold change stays CPU-only (valid while segment output is unchanged).
    prev_asr = None
    if reasr and bool(cfg.get("quality.reasr_reuse", False)) and out_path.exists():
        prev_asr = {
            c["clip_id"]: (c["cer"], c["asr_text"])
            for c in read_jsonl(out_path)
            if "asr_text" in c
        }
        log.info("quality: reusing stored re-ASR results for %d clips", len(prev_asr))
    jobs = _resolve_jobs(cfg, reasr)

    if reasr and not (prev_asr is not None and all(c["clip_id"] in prev_asr for c in clips)):
        # Load the model up front so worker threads share one instance instead
        # of racing the cache.
        from .transcribe import _get_model

        _get_model(cfg)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                _check_clip, cfg, clip,
                min_dur, max_dur, max_cer, review_cer, min_snr, max_sil, reasr,
                edge, prev_asr,
            )
            for clip in clips
        ]
        progress = tqdm(
            as_completed(futures), total=len(futures),
            desc=f"quality: {book_id}", unit="clip",
        )
        for future in progress:
            future.result()

    # Re-apply stored human verdicts so they survive --force re-runs.
    review_dir = work_book_dir(cfg, book_id) / "review"
    review.apply_verdicts(clips, review.load_verdicts(review_dir / "verdicts.csv"))

    # Workers mutate the ordered `clips` list in place, so output order matches
    # the serial run regardless of completion order.
    write_jsonl(out_path, clips)
    _write_flagged(review_dir, clips)

    counts = {k: sum(1 for c in clips if c["status"] == k) for k in ("pass", "review", "reject")}
    log.info("quality: %s — pass=%d review=%d reject=%d",
             book_id, counts["pass"], counts["review"], counts["reject"])
    return clips


def _write_flagged(review_dir, clips: list[dict]) -> None:
    """Flagged-for-review export (everything not a clean pass)."""
    review_dir.mkdir(parents=True, exist_ok=True)
    flagged = [c for c in clips if c["status"] != "pass"]
    with (review_dir / "flagged.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "status", "flags", "duration", "snr_db",
                         "silence_ratio", "cer", "edge_lead_ms", "edge_tail_ms",
                         "text", "wav"])
        for c in flagged:
            writer.writerow([
                c["clip_id"], c["status"], "|".join(c["flags"]), c["duration"],
                c.get("snr_db"), c.get("silence_ratio"), c.get("cer"),
                c.get("edge_lead_ms"), c.get("edge_tail_ms"),
                c["text"], c["wav"],
            ])
