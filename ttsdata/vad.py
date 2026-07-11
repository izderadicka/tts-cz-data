"""Lightweight audio analysis: framing, RMS, silence detection, boundary snapping.

Two detectors are available for boundary snapping:

- **silero** (default): the silero-vad ONNX model bundled with faster-whisper.
  :func:`speech_regions` runs it once per chapter; the pure
  :func:`snap_start_regions` / :func:`snap_end_regions` helpers then snap cut
  points to region boundaries. Neural VAD catches low-energy unvoiced
  consonants (final stop bursts, fricatives) that an energy threshold misses.
- **energy**: numpy-only frame-RMS fallback (:func:`snap_start` /
  :func:`snap_end`), used when faster-whisper isn't installed. The energy
  helpers also back the quality stage's silence/SNR metrics.
"""

from __future__ import annotations

import bisect

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


def edge_noise_ms(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: int = 10,
    strong_rel: float = 0.30,
    weak_rel: float = 0.05,
    max_centroid_hz: float = 900.0,
    min_gap_ms: float = 0.0,
) -> float:
    """Duration (ms) of speaker noise fused to the clip's leading edge.

    Audiobook narrators sometimes emit a short grunt/creak/thump right before
    (or after) an utterance. Silero counts it as speech, so it survives
    segmentation; whole-clip metrics don't see it. This scans the frames before
    the first "strong" frame (>= ``strong_rel`` of the clip's 95th-pct RMS) for
    the longest weak run (``weak_rel``..``strong_rel``) that rises out of
    silence and whose median spectral centroid is below ``max_centroid_hz``
    (voiced grunts are low-frequency; breaths/fricatives are not). Returns 0.0
    when no run qualifies. The caller decides how many ms are suspicious —
    short runs are often legitimate prevoicing of B/D-initial words.

    Call on ``samples[::-1]`` to check the trailing edge; there, pass
    ``min_gap_ms`` > 0 so only runs *detached* from the last word by silence
    qualify (a contiguous weak tail is just natural voiced decay).
    """
    samples = to_mono(samples).astype(np.float64)
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames < 2:
        return 0.0
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    p95 = float(np.percentile(rms, 95))
    strong = rms >= strong_rel * p95
    if not strong.any():
        return 0.0
    i0 = int(np.argmax(strong))
    floor = max(weak_rel * p95, 1e-4)
    min_gap = int(round(min_gap_ms / frame_ms))

    window = np.hanning(frame_len)
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)

    def _centroid(i: int) -> float:
        mag = np.abs(np.fft.rfft(frames[i] * window))
        total = mag.sum()
        return float((mag * freqs).sum() / total) if total > 0 else 0.0

    best = 0.0
    run_start: int | None = None
    for f in range(i0 + 1):
        weak = f < i0 and rms[f] >= floor
        if weak and run_start is None:
            run_start = f
        elif not weak and run_start is not None:
            # A genuine artifact is an island rising out of silence (the
            # segment stage pads ~100 ms before the cut, so silence is there
            # to see); a run touching clip start is a legit onset ramp.
            preceded_by_silence = run_start > 0 and rms[run_start - 1] < floor
            # ``min_gap`` silent frames must separate the run from the strong
            # onset; a run ending at i0 itself has gap 0 (fused to the word).
            gap_ok = i0 - f >= min_gap and all(rms[g] < floor for g in range(f, f + min_gap))
            if preceded_by_silence and gap_ok:
                centroid = float(np.median([_centroid(i) for i in range(run_start, f)]))
                if centroid < max_centroid_hz:
                    best = max(best, (f - run_start) * frame_ms)
            run_start = None
    return best


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


def speech_regions(
    audio: np.ndarray,
    sr: int = 16000,
    *,
    threshold: float = 0.5,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 30,
) -> list[tuple[float, float]]:
    """Detect speech regions with silero VAD; returns [(start_s, end_s), ...].

    Uses the silero ONNX model bundled with faster-whisper (thread-safe, its
    RNN state is per-call). faster-whisper's defaults (2 s min silence, 400 ms
    pad) target long-form ASR chunking, so the clip-cutting values are set
    explicitly here.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        raise ImportError(
            "silero VAD needs faster-whisper; install ttsdata[asr] "
            "or set segment.vad: energy"
        ) from e

    opts = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    chunks = get_speech_timestamps(to_mono(audio).astype(np.float32), opts, sampling_rate=sr)
    return [(c["start"] / sr, c["end"] / sr) for c in chunks]


def snap_end_regions(
    regions: list[tuple[float, float]],
    t: float,
    *,
    search_back_s: float = 0.24,
    search_fwd_s: float = 0.8,
) -> float:
    """Snap a segment end ``t`` to the nearest speech-region end.

    Inside a region, move to its end unless that lies beyond ``search_fwd_s``
    (speech continues — never cut far into it, keep ``t``). In a gap, pull back
    to the previous region's end if it's within ``search_back_s``.
    """
    i = bisect.bisect_right([r[0] for r in regions], t) - 1
    if i < 0:
        return t
    end = regions[i][1]
    if t < end:  # inside region i
        return end if end <= t + search_fwd_s else t
    return end if end >= t - search_back_s else t


def snap_start_regions(
    regions: list[tuple[float, float]],
    t: float,
    *,
    search_back_s: float = 0.24,
    search_fwd_s: float = 0.8,
) -> float:
    """Mirror of :func:`snap_end_regions`: snap a segment start to the
    containing region's start, or forward to the next region's start."""
    i = bisect.bisect_right([r[0] for r in regions], t) - 1
    if i >= 0 and t < regions[i][1]:  # inside region i
        start = regions[i][0]
        return start if start >= t - search_fwd_s else t
    if i + 1 < len(regions):  # in a gap (or before all regions)
        nxt = regions[i + 1][0]
        return nxt if nxt <= t + search_back_s else t
    return t
