"""Stage 1 — text preparation.

Reads the book's plain text, cleans it, splits into Czech sentences, and stores
both a human-readable label (``text``) and a fully spoken-out form
(``normalized``) plus alignment ``tokens`` for each sentence.
"""

from __future__ import annotations

import logging

from ..config import Config
from ..manifest import read_jsonl, write_jsonl
from ..text import normalize_cs, sentences_cs
from ..workspace import raw_book_dir, stage_manifest

log = logging.getLogger(__name__)


def _read_book_text(cfg: Config, book_id: str) -> str:
    book_path = raw_book_dir(cfg, book_id) / "book.txt"
    if not book_path.exists():
        raise FileNotFoundError(f"Missing book text: {book_path}")
    return book_path.read_text(encoding="utf-8")


def run(cfg: Config, book_id: str, force: bool = False) -> list[dict]:
    out_path = stage_manifest(cfg, book_id, "text_prep")
    if out_path.exists() and not force:
        log.info("text_prep: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    raw_text = _read_book_text(cfg, book_id)
    sentences = sentences_cs.split_sentences(raw_text)

    records: list[dict] = []
    for order, sentence in enumerate(sentences):
        text = normalize_cs.clean_text(sentence)
        normalized = normalize_cs.normalize(sentence)
        tokens = normalize_cs.tokenize(normalized)
        if not tokens:
            continue  # punctuation-only fragment
        records.append(
            {
                "book_id": book_id,
                "sent_id": f"{book_id}_s{order:05d}",
                "order": order,
                "text": text,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

    write_jsonl(out_path, records)
    log.info("text_prep: %s — %d sentences", book_id, len(records))
    return records
