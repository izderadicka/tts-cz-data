"""Tests for the ASR-bridge aligner using synthetic word timings."""

from ttsdata.config import load_config
from ttsdata.stages.align.asr_bridge import AsrBridgeAligner
from ttsdata.text.normalize_cs import tokenize


def _words(pairs):
    """Build ASR word records from (word, start, end) tuples."""
    return [{"word": w, "start": s, "end": e, "prob": 0.9} for w, s, e in pairs]


def _sentence(book_id, order, text):
    return {
        "book_id": book_id,
        "sent_id": f"{book_id}_s{order:05d}",
        "order": order,
        "text": text,
        "normalized": text,
        "tokens": tokenize(text),
    }


def test_aligner_maps_sentences_to_times():
    cfg = load_config()
    # Two book sentences.
    sentences = [
        _sentence("b", 0, "Ahoj světe"),
        _sentence("b", 1, "Jak se máš"),
    ]
    # One chapter whose ASR words cover both sentences (with diacritic noise).
    chapter = {
        "chapter_id": "ch01",
        "order": 0,
        "words": _words([
            ("Ahoj", 0.0, 0.5),
            ("svete", 0.5, 1.0),   # ASR dropped the diacritic on "světe"
            ("jak", 1.5, 1.8),
            ("se", 1.8, 2.0),
            ("máš", 2.0, 2.4),
        ]),
    }

    segs = AsrBridgeAligner().align(cfg, [chapter], sentences)
    assert len(segs) == 2

    s0, s1 = sorted(segs, key=lambda s: s.start)
    assert s0.text == "Ahoj světe"
    assert s0.start == 0.0 and s0.end == 1.0
    assert s0.score == 1.0
    assert s0.chapter_id == "ch01"

    assert s1.text == "Jak se máš"
    assert s1.start == 1.5 and s1.end == 2.4
    assert s1.score == 1.0


def test_aligner_partial_match_scores_below_one():
    cfg = load_config()
    sentences = [_sentence("b", 0, "jedna dva tri ctyri")]
    chapter = {
        "chapter_id": "ch01",
        "order": 0,
        "words": _words([("jedna", 0.0, 0.4), ("dva", 0.4, 0.8)]),  # only 2/4
    }
    segs = AsrBridgeAligner().align(cfg, [chapter], sentences)
    assert len(segs) == 1
    assert segs[0].score == 0.5


def test_aligner_skips_unmatched_sentence():
    cfg = load_config()
    sentences = [
        _sentence("b", 0, "tahle veta zazni"),
        _sentence("b", 1, "tahle veta chybi v nahravce"),
    ]
    chapter = {
        "chapter_id": "ch01",
        "order": 0,
        "words": _words([("tahle", 0.0, 0.3), ("veta", 0.3, 0.6), ("zazni", 0.6, 0.9)]),
    }
    segs = AsrBridgeAligner().align(cfg, [chapter], sentences)
    # The second sentence has no matching audio -> not emitted (or score ~0).
    assert any(s.seg_id == "b_s00000" for s in segs)
    matched_ids = {s.seg_id for s in segs if s.score >= 0.5}
    assert "b_s00001" not in matched_ids
