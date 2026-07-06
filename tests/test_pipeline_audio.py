"""Integration test for segment -> quality -> export using synthetic audio.

We synthesise a chapter WAV with two tone bursts separated by silence and hand
the stages a matching align manifest, so the audio-dependent half of the
pipeline runs end to end without ffmpeg or ASR models.
"""

import numpy as np
import soundfile as sf

from ttsdata.config import load_config
from ttsdata.manifest import read_jsonl, write_jsonl
from ttsdata.stages import export, quality, segment
from ttsdata.workspace import stage_manifest


def _make_chapter_wav(path, sr=22050):
    """Two 1.5s tone bursts at [0.5,2.0] and [3.0,4.5], silence elsewhere."""
    n = int(5.0 * sr)
    audio = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / sr
    for lo, hi, f in [(0.5, 2.0, 220.0), (3.0, 4.5, 330.0)]:
        mask = (t >= lo) & (t < hi)
        audio[mask] = 0.3 * np.sin(2 * np.pi * f * t[mask]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def _override_paths(tmp_path):
    cfg = load_config()
    cfg._data["paths"]["work"] = str(tmp_path / "work")
    cfg._data["paths"]["dataset"] = str(tmp_path / "dataset")
    cfg._data["paths"]["raw"] = str(tmp_path / "raw")
    # Pin off regardless of the config default: these tests use synthetic tones
    # and must run without ASR models.
    cfg._data["quality"]["reasr_check"] = False
    cfg.repo_root = tmp_path  # paths already absolute
    return cfg


def test_segment_quality_export(tmp_path):
    cfg = _override_paths(tmp_path)
    book_id = "synt"

    master = tmp_path / "work" / book_id / "00_ingest" / "master" / "ch01.wav"
    _make_chapter_wav(master)

    # ingest manifest (segment reads master_wav + speaker_id from here)
    write_jsonl(stage_manifest(cfg, book_id, "ingest"), [
        {"book_id": book_id, "speaker_id": book_id, "chapter_id": "ch01",
         "order": 0, "master_wav": str(master), "duration": 5.0},
    ])
    # align manifest: two segments matching the two tone bursts
    write_jsonl(stage_manifest(cfg, book_id, "align"), [
        {"book_id": book_id, "chapter_id": "ch01", "seg_id": "synt_s00000",
         "text": "První věta.", "normalized": "první věta", "start": 0.5,
         "end": 2.0, "score": 1.0},
        {"book_id": book_id, "chapter_id": "ch01", "seg_id": "synt_s00001",
         "text": "Druhá věta.", "normalized": "druhá věta", "start": 3.0,
         "end": 4.5, "score": 1.0},
    ])

    clips = segment.run(cfg, book_id, force=True)
    assert len(clips) == 2
    assert all(c["duration"] > 1.0 for c in clips)

    qclips = quality.run(cfg, book_id, force=True)
    # tone bursts are loud and non-silent -> should pass duration/silence checks
    assert {c["status"] for c in qclips} <= {"pass", "review"}
    assert all("duration" not in c["flags"] for c in qclips)

    stats = export.run(cfg, book_id, force=True)
    dataset_dir = tmp_path / "dataset" / book_id
    assert (dataset_dir / "metadata.csv").exists()
    assert (dataset_dir / "PIPER.md").exists()
    meta = (dataset_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert len(meta) == stats["clips"]
    assert "|" in meta[0]


def test_flagged_csv_written(tmp_path):
    cfg = _override_paths(tmp_path)
    book_id = "synt2"
    master = tmp_path / "work" / book_id / "00_ingest" / "master" / "ch01.wav"
    _make_chapter_wav(master)
    write_jsonl(stage_manifest(cfg, book_id, "ingest"), [
        {"book_id": book_id, "speaker_id": book_id, "chapter_id": "ch01",
         "order": 0, "master_wav": str(master), "duration": 5.0},
    ])
    # A too-short segment (0.2s) should be rejected on duration.
    write_jsonl(stage_manifest(cfg, book_id, "align"), [
        {"book_id": book_id, "chapter_id": "ch01", "seg_id": "x",
         "text": "krátké", "normalized": "krátké", "start": 0.6, "end": 0.8,
         "score": 1.0},
    ])
    segment.run(cfg, book_id, force=True)
    qclips = quality.run(cfg, book_id, force=True)
    assert any(c["status"] == "reject" for c in qclips)
    flagged = tmp_path / "work" / book_id / "review" / "flagged.csv"
    assert flagged.exists()
