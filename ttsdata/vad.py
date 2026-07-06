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


def snap_to_silence(samples: np.ndarray, sr: int, t: float, window_s: float) -> float:
    """Move a cut time ``t`` to the quietest spot within +/- ``window_s``.

    Cutting at a local energy minimum avoids clipping words mid-syllable.
    Only the window around ``t`` is analysed, so this stays cheap even when
    ``samples`` is a whole chapter.
    """
    # Pad the slice by one frame so silence starting right at the window edge
    # still yields a fully-quiet frame regardless of framing alignment.
    frame_len = max(1, int(sr * FRAME_MS / 1000))
    lo_sample = max(0, int(round((t - window_s) * sr)) - frame_len)
    hi_sample = min(len(samples), int(round((t + window_s) * sr)) + frame_len)
    if hi_sample <= lo_sample:
        return t
    rms, frame_len = frame_rms(samples[lo_sample:hi_sample], sr)
    best = int(np.argmin(rms))
    return (lo_sample + best * frame_len) / sr
