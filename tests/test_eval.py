"""Tests for the aligner comparison harness (audio-free metrics)."""

from ttsdata.eval.compare_aligners import basic_metrics, format_report


def test_basic_metrics():
    segments = [
        {"chapter_id": "ch01", "start": 0.0, "end": 2.0, "score": 1.0},
        {"chapter_id": "ch01", "start": 3.0, "end": 6.0, "score": 0.8},
    ]
    durations = {"ch01": 10.0}
    m = basic_metrics(segments, durations)
    assert m["segments"] == 2
    assert m["audio_seconds"] == 5.0
    assert m["retained_pct"] == 50.0
    assert m["mean_score"] == 0.9
    assert m["dur_max"] == 3.0


def test_format_report_renders_strategies():
    report = {
        "asr_bridge": basic_metrics(
            [{"chapter_id": "c", "start": 0, "end": 2, "score": 1.0}], {"c": 4.0}
        ),
        "forced": basic_metrics(
            [{"chapter_id": "c", "start": 0, "end": 3, "score": 0.9}], {"c": 4.0}
        ),
    }
    text = format_report(report)
    assert "asr_bridge" in text
    assert "forced" in text
    assert "retained_pct" in text
