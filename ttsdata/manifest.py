"""JSONL manifest I/O — the backbone connecting pipeline stages.

Each stage reads the previous stage's manifest (a list of JSON records, one per
line) and writes its own. Keeping records as plain dicts keeps stages decoupled
and the on-disk format trivially inspectable (``head -n1 file.jsonl | jq``).

Canonical record shapes (documented, not enforced) as they grow through stages:

  Chapter (stage 0 ``ingest``):
    {book_id, speaker_id, chapter_id, order, src_path,
     asr_wav, master_wav, duration, sample_rate_master, codec}

  Sentence (stage 1 ``text_prep``, per book, separate file):
    {book_id, sent_id, order, text, normalized, tokens}

  Word (stage 2 ``transcribe``, embedded in chapter record as ``words``):
    {start, end, word, prob}

  Segment (stage 3 ``align``):
    {book_id, chapter_id, seg_id, text, normalized, start, end, score}

  Clip (stage 4 ``segment`` onward):
    Segment + {clip_id, wav, duration, speaker_id,
               status, cer, snr_db, flags}   # status: pass|review|reject
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file (empty list if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream records one at a time (for large manifests)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write records to a JSONL file (parent dirs created). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def write_json(path: str | Path, obj: Any) -> None:
    """Write a single JSON document (used for stats/reports)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
