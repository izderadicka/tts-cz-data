"""Lightweight audio analysis: framing, RMS, silence detection, boundary snapping.

Energy-based and dependency-free (numpy only) so the pipeline runs on CPU with
no extra installs. ``silero-vad`` can be plugged in later as a higher-quality
option behind the same helpers; the energy detector is the default.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 20


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        return samples.mean(axis=1)
    return samples


def frame_rms(samples: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> tuple[np.ndarray, int]:
    """Return (rms_per_frame, frame_len_samples)."""
    samples = to_mono(samples).astype(np.float64)
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(samples) // frame_len)
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)
    return rms, frame_len


def silence_threshold(rms: np.ndarray, rel: float = 0.15) -> float:
    """Adaptive threshold: a fraction of the loud (95th-pct) level."""
    loud = np.percentile(rms, 95)
    return max(loud * rel, 1e-4)


def silence_ratio(samples: np.ndarray, sr: int) -> float:
    rms, _ = frame_rms(samples, sr)
    thr = silence_threshold(rms)
    return float(np.mean(rms < thr))


def estimate_snr_db(samples: np.ndarray, sr: int) -> float:
    """Rough SNR: speech-frame power vs silence-frame power, in dB."""
    rms, _ = frame_rms(samples, sr)
    thr = silence_threshold(rms)
    speech = rms[rms >= thr]
    noise = rms[rms < thr]
    if speech.size == 0:
        return 0.0
    noise_p = float(np.mean(noise**2)) if noise.size else 1e-8
    speech_p = float(np.mean(speech**2))
    return 10.0 * np.log10(speech_p / max(noise_p, 1e-8))


def snap_end(
    rms: np.ndarray,
    frame_len: int,
    sr: int,
    thr: float,
    t: float,
    *,
    search_back_s: float = 0.24,
    search_fwd_s: float = 0.8,
    min_pause_s: float = 0.3,
) -> float:
    """Move a segment end ``t`` to just after the last speech before a real pause.

    ASR end timestamps run short and a naive local energy minimum lands on
    intra-word dips (e.g. the ~160 ms stop closure in "Pippi"), clipping the
    final syllable. Instead, walk forward through precomputed frame RMS and cut
    after the last frame above ``thr`` once a silence run of ``min_pause_s``
    (too long for a stop closure) confirms the sentence really ended. Without
    such a pause in the window, return ``t`` unchanged — never cut into speech.
    """
    f_lo = max(0, int((t - search_back_s) * sr / frame_len))
    f_hi = min(len(rms), int((t + search_fwd_s) * sr / frame_len) + 1)
    min_pause = max(1, int(min_pause_s * sr / frame_len))
    last_speech = None
    run = 0
    for f in range(f_lo, f_hi):
        if rms[f] >= thr:
            last_speech = f
            run = 0
        else:
            run += 1
            if run >= min_pause:
                if last_speech is None:
                    return t
                return (last_speech + 1) * frame_len / sr
    return t


def snap_start(
    rms: np.ndarray,
    frame_len: int,
    sr: int,
    thr: float,
    t: float,
    *,
    search_back_s: float = 0.24,
    search_fwd_s: float = 0.8,
    min_pause_s: float = 0.3,
) -> float:
    """Mirror of :func:`snap_end`: move a segment start to just before the
    first speech after a real pause, walking backwards from ``t``."""
    f_hi = min(len(rms) - 1, int((t + search_back_s) * sr / frame_len))
    f_lo = max(0, int((t - search_fwd_s) * sr / frame_len))
    min_pause = max(1, int(min_pause_s * sr / frame_len))
    first_speech = None
    run = 0
    for f in range(f_hi, f_lo - 1, -1):
        if rms[f] >= thr:
            first_speech = f
            run = 0
        else:
            run += 1
            if run >= min_pause:
                if first_speech is None:
                    return t
                return first_speech * frame_len / sr
    return t
