"""Tests for the review verdict store and apply semantics (ttsdata.review)."""

import copy

import pytest

from ttsdata import review


def _clips():
    return [
        {"clip_id": "a_0001", "status": "pass", "flags": []},
        {"clip_id": "a_0002", "status": "review", "flags": ["edge_noise"]},
        {"clip_id": "a_0003", "status": "review", "flags": ["edge_noise"]},
        {"clip_id": "a_0004", "status": "reject", "flags": ["duration"]},
    ]


def test_approve_lifts_review_to_pass():
    clips = _clips()
    counts = review.apply_verdicts(clips, {"a_0002": "approve"})
    assert clips[1]["status"] == "pass"
    assert clips[1]["verdict"] == "approve"
    assert clips[1]["flags"] == ["edge_noise"]  # provenance kept
    assert counts == {"approved": 1, "rejected": 0, "stale": 0}


def test_approve_does_not_lift_hard_reject():
    clips = _clips()
    review.apply_verdicts(clips, {"a_0004": "approve"})
    assert clips[3]["status"] == "reject"


def test_reject_forces_any_status():
    clips = _clips()
    counts = review.apply_verdicts(clips, {"a_0001": "reject", "a_0003": "reject"})
    assert clips[0]["status"] == "reject"
    assert clips[2]["status"] == "reject"
    assert counts["rejected"] == 2


def test_stale_verdict_counted_and_clips_untouched():
    clips = _clips()
    before = copy.deepcopy(clips)
    counts = review.apply_verdicts(clips, {"gone_0001": "reject"})
    assert counts["stale"] == 1
    assert clips == before


def test_apply_is_idempotent():
    clips = _clips()
    verdicts = {"a_0002": "approve", "a_0003": "reject"}
    review.apply_verdicts(clips, verdicts)
    snapshot = copy.deepcopy(clips)
    review.apply_verdicts(clips, verdicts)
    assert clips == snapshot


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "review" / "verdicts.csv"
    review.save_verdict(path, "b_0002", "reject", "edge_noise|low_snr")
    review.save_verdict(path, "b_0001", "approve", "edge_noise")
    assert review.load_verdicts(path) == {"b_0001": "approve", "b_0002": "reject"}
    # sorted by clip_id for diff-friendliness
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "clip_id,verdict,flags"
    assert lines[1].startswith("b_0001,") and lines[2].startswith("b_0002,")


def test_save_verdict_overwrites_last_wins(tmp_path):
    path = tmp_path / "verdicts.csv"
    review.save_verdict(path, "c_0001", "reject")
    review.save_verdict(path, "c_0001", "approve")
    assert review.load_verdicts(path) == {"c_0001": "approve"}


def test_save_verdict_rejects_unknown_verdict(tmp_path):
    with pytest.raises(ValueError):
        review.save_verdict(tmp_path / "verdicts.csv", "d_0001", "maybe")


def test_load_verdicts_missing_file(tmp_path):
    assert review.load_verdicts(tmp_path / "nope.csv") == {}
