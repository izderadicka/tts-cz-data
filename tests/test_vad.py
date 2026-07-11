"""Tests for silence-aware boundary snapping (vad.snap_start / snap_end)."""

import numpy as np
import pytest

from ttsdata import vad

# Frame geometry: sr=1000, frame_len=20 -> one frame = 20 ms, like production.
SR = 1000
FRAME_LEN = 20
THR = 0.5


def _rms(pattern: list[tuple[float, float]]) -> np.ndarray:
    """Build an RMS array from (duration_s, level) spans."""
    out: list[float] = []
    for dur, level in pattern:
        out.extend([level] * int(dur * SR / FRAME_LEN))
    return np.array(out)


def test_snap_end_skips_intra_word_dip():
    # speech | 160ms stop-closure dip | final syllable | real pause
    # (the "Pippi" case: the dip must not be mistaken for the sentence end)
    rms = _rms([(1.0, 1.0), (0.16, 0.0), (0.14, 1.0), (1.0, 0.0)])
    end = vad.snap_end(rms, FRAME_LEN, SR, THR, t=1.0)
    assert abs(end - 1.30) < 0.021  # after the final syllable, not at the dip


def test_snap_end_never_cuts_backward_into_speech():
    # speech runs past the aligned end t; silence only from 1.1s
    rms = _rms([(1.1, 1.0), (1.0, 0.0)])
    end = vad.snap_end(rms, FRAME_LEN, SR, THR, t=1.0)
    assert end >= 1.1 - 0.021


def test_snap_end_unchanged_without_pause():
    # continuous speech through the whole window -> keep aligned end
    rms = _rms([(3.0, 1.0)])
    assert vad.snap_end(rms, FRAME_LEN, SR, THR, t=1.5) == 1.5


def test_snap_end_trims_when_already_in_silence():
    # speech ends before t; pause confirmed -> cut right after last speech
    rms = _rms([(0.9, 1.0), (2.0, 0.0)])
    end = vad.snap_end(rms, FRAME_LEN, SR, THR, t=1.0)
    assert abs(end - 0.9) < 0.021


def test_snap_start_mirrors_over_leading_dip():
    # pause | first syllable | 160ms closure dip | speech ; aligned start at 1.0
    rms = _rms([(0.7, 0.0), (0.14, 1.0), (0.16, 0.0), (1.5, 1.0)])
    start = vad.snap_start(rms, FRAME_LEN, SR, THR, t=1.0)
    assert abs(start - 0.70) < 0.021  # back to the true word onset


def test_snap_start_unchanged_without_pause():
    rms = _rms([(3.0, 1.0)])
    assert vad.snap_start(rms, FRAME_LEN, SR, THR, t=1.5) == 1.5


# --- silero region snapping (pure helpers, no model needed) ---

REGIONS = [(0.5, 2.0), (3.0, 5.0), (10.0, 20.0)]


def test_snap_end_regions_extends_to_region_end():
    # aligned end lands mid-region (e.g. ASR ran short of a final consonant)
    assert vad.snap_end_regions(REGIONS, t=1.8) == 2.0


def test_snap_end_regions_unchanged_when_speech_continues():
    # region end is beyond the forward window -> never cut far into speech
    assert vad.snap_end_regions(REGIONS, t=11.0) == 11.0


def test_snap_end_regions_trims_back_from_gap():
    # aligned end fell into the pause just after a region
    assert vad.snap_end_regions(REGIONS, t=2.1) == 2.0


def test_snap_end_regions_unchanged_deep_in_gap():
    # too far past the region end to safely pull back
    assert vad.snap_end_regions(REGIONS, t=2.6) == 2.6


def test_snap_end_regions_before_all_regions():
    assert vad.snap_end_regions(REGIONS, t=0.2) == 0.2


def test_snap_start_regions_extends_to_region_start():
    assert vad.snap_start_regions(REGIONS, t=3.4) == 3.0


def test_snap_start_regions_unchanged_when_speech_precedes():
    # region start is beyond the backward window -> keep aligned start
    assert vad.snap_start_regions(REGIONS, t=15.0) == 15.0


def test_snap_start_regions_advances_from_gap():
    # aligned start fell into the pause just before a region
    assert vad.snap_start_regions(REGIONS, t=2.8) == 3.0


def test_snap_start_regions_unchanged_deep_in_gap():
    assert vad.snap_start_regions(REGIONS, t=2.2) == 2.2


def test_snap_regions_empty():
    assert vad.snap_end_regions([], t=1.0) == 1.0
    assert vad.snap_start_regions([], t=1.0) == 1.0


def test_speech_regions_silent_audio_has_none():
    pytest.importorskip("faster_whisper")
    silence = np.zeros(16000 * 2, dtype=np.float32)
    assert vad.speech_regions(silence) == []


# --- edge_noise_ms (leading/trailing speaker-noise detection) ---

EDGE_SR = 16000


def _tone(dur_s: float, amp: float, freq_hz: float) -> np.ndarray:
    t = np.arange(int(dur_s * EDGE_SR)) / EDGE_SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _spans(*parts: tuple[float, float, float]) -> np.ndarray:
    """Concatenate (duration_s, amplitude, freq_hz) spans; amp 0 = silence."""
    return np.concatenate([_tone(d, a, f) for d, a, f in parts])


SPEECH = (1.0, 0.5, 1500.0)  # strong high-centroid "speech"
GRUNT_AMP = 0.06             # weak relative to 0.5-amp speech


def test_edge_noise_detects_low_freq_lead():
    # silence | 60ms grunt | speech — the reference-clip shape
    audio = _spans((0.3, 0.0, 0.0), (0.06, GRUNT_AMP, 150.0), SPEECH)
    assert vad.edge_noise_ms(audio, EDGE_SR) >= 50


def test_edge_noise_clean_onset():
    audio = _spans((0.3, 0.0, 0.0), SPEECH)
    assert vad.edge_noise_ms(audio, EDGE_SR) == 0.0


def test_edge_noise_short_prevoicing_stays_below_flag_threshold():
    # 20ms voiced lead: detected, but short enough for the caller's min_ms gate
    audio = _spans((0.3, 0.0, 0.0), (0.02, GRUNT_AMP, 150.0), SPEECH)
    assert vad.edge_noise_ms(audio, EDGE_SR) < 40


def test_edge_noise_high_centroid_lead_ignored():
    # a breath/fricative-like lead (white noise, centroid >> 900 Hz)
    rng = np.random.default_rng(0)
    breath = (GRUNT_AMP * rng.standard_normal(int(0.06 * EDGE_SR))).astype(np.float32)
    audio = np.concatenate([_tone(0.3, 0.0, 0.0), breath, _tone(*SPEECH)])
    assert vad.edge_noise_ms(audio, EDGE_SR) == 0.0


def test_edge_noise_requires_pre_silence():
    # gradual low-freq ramp into the word: weak span not preceded by silence
    t = np.arange(int(0.3 * EDGE_SR)) / EDGE_SR
    ramp = (np.linspace(0.04, 0.5, t.size) * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    audio = np.concatenate([ramp, _tone(*SPEECH)])
    assert vad.edge_noise_ms(audio, EDGE_SR) == 0.0


def test_edge_noise_tail_island_via_reverse():
    # speech | 100ms silence | 60ms grunt | silence — detached trailing island
    audio = _spans(SPEECH, (0.1, 0.0, 0.0), (0.06, GRUNT_AMP, 150.0), (0.2, 0.0, 0.0))
    assert vad.edge_noise_ms(audio[::-1], EDGE_SR, min_gap_ms=30) >= 50


def test_edge_noise_tail_ignores_natural_decay():
    # speech with a contiguous weak voiced decay, then silence
    audio = _spans(SPEECH, (0.08, GRUNT_AMP, 150.0), (0.3, 0.0, 0.0))
    assert vad.edge_noise_ms(audio[::-1], EDGE_SR, min_gap_ms=30) == 0.0


def test_edge_noise_all_silence():
    assert vad.edge_noise_ms(np.zeros(EDGE_SR), EDGE_SR) == 0.0
