"""Tests for config loading and manifest round-trip."""

from ttsdata.config import Config, load_config
from ttsdata.manifest import read_jsonl, write_jsonl, write_json


def test_default_config_loads():
    cfg = load_config()
    assert cfg.get("language") == "cs"
    assert cfg.get("audio.asr_sample_rate") == 16000
    assert cfg.get("missing.key", "fallback") == "fallback"


def test_config_deep_merge():
    base = Config({"a": {"b": 1, "c": 2}, "d": 3})
    # _deep_merge is exercised via load_config; here we check dotted access.
    assert base.get("a.b") == 1
    assert base.get("a.c") == 2
    assert base.get("d") == 3


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "m.jsonl"
    records = [
        {"id": "a", "text": "ahoj světe", "n": 1},
        {"id": "b", "text": "příliš žluťoučký kůň", "n": 2},
    ]
    assert write_jsonl(path, records) == 2
    loaded = read_jsonl(path)
    assert loaded == records
    # unicode (Czech diacritics) survives the round trip
    assert loaded[1]["text"] == "příliš žluťoučký kůň"


def test_read_missing_manifest(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_write_json(tmp_path):
    path = tmp_path / "stats.json"
    write_json(path, {"hours": 1.5, "clips": 10})
    assert path.exists()
