"""Tests for silence-aware boundary snapping (vad.snap_start / snap_end)."""

import numpy as np

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
