"""ASR-bridge aligner (Strategy A).

Idea: the book text is clean but has no timing; the ASR transcript is noisy but
time-stamped. We sequence-align the two token streams, then read each book
sentence's audio time span off the ASR words it matched. The **book** sentence
stays the label (we never use the noisy ASR as the target).

This is robust to narrator skips/edits and to skipped front/back matter, because
the alignment simply leaves unmatched stretches unmatched.

Algorithm:
  1. Flatten ASR words across chapters (in order) -> tokens tagged (chapter, t0, t1).
  2. Flatten book sentences -> tokens tagged with sentence id.
  3. ``difflib.SequenceMatcher`` (diacritic-folded tokens) finds matching blocks.
  4. Per sentence: gather matched ASR token times -> [start, end]; score =
     matched_tokens / sentence_tokens.

Complexity is difflib's (≈O(n·m) worst case); fine for chapter/book scales. For
very large books this can be chunked later, but the baseline keeps it simple.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from ...config import Config
from .base import Segment

log = logging.getLogger(__name__)


def _fold(token: str) -> str:
    """Lowercase and strip diacritics so ASR/book minor mismatches still align."""
    decomposed = unicodedata.normalize("NFKD", token.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _flatten_asr(chapters: list[dict]) -> tuple[list[str], list[dict]]:
    """Return (folded tokens, meta) where meta[i] = {chapter_id, start, end}."""
    from ...text.normalize_cs import tokenize

    tokens: list[str] = []
    meta: list[dict] = []
    for ch in sorted(chapters, key=lambda c: c.get("order", 0)):
        for w in ch.get("words", []):
            for tok in tokenize(w["word"]):
                tokens.append(_fold(tok))
                meta.append(
                    {"chapter_id": ch["chapter_id"], "start": w["start"], "end": w["end"]}
                )
    return tokens, meta


def _flatten_book(sentences: list[dict]) -> tuple[list[str], list[int]]:
    """Return (folded tokens, owner) where owner[i] = sentence index."""
    tokens: list[str] = []
    owner: list[int] = []
    for si, sent in enumerate(sentences):
        for tok in sent["tokens"]:
            tokens.append(_fold(tok))
            owner.append(si)
    return tokens, owner


class AsrBridgeAligner:
    def align(
        self, cfg: Config, chapters: list[dict], sentences: list[dict]
    ) -> list[Segment]:
        asr_tokens, asr_meta = _flatten_asr(chapters)
        book_tokens, book_owner = _flatten_book(sentences)
        if not asr_tokens or not book_tokens:
            log.warning("align: empty ASR or book tokens; nothing to align")
            return []

        # Per-sentence accumulators.
        n = len(sentences)
        matched = [0] * n
        starts: list[list[float]] = [[] for _ in range(n)]
        ends: list[list[float]] = [[] for _ in range(n)]
        chapters_hit: list[Counter] = [Counter() for _ in range(n)]

        sm = SequenceMatcher(None, book_tokens, asr_tokens, autojunk=False)
        for bi, ai, size in sm.get_matching_blocks():
            for k in range(size):
                si = book_owner[bi + k]
                m = asr_meta[ai + k]
                matched[si] += 1
                starts[si].append(m["start"])
                ends[si].append(m["end"])
                chapters_hit[si][m["chapter_id"]] += 1

        segments: list[Segment] = []
        for si, sent in enumerate(sentences):
            total = len(sent["tokens"])
            if matched[si] == 0 or total == 0:
                continue
            start = min(starts[si])
            end = max(max(ends[si]), start)  # guard against inverted spans
            chapter_id = chapters_hit[si].most_common(1)[0][0]
            segments.append(
                Segment(
                    book_id=sent["book_id"],
                    chapter_id=chapter_id,
                    seg_id=sent["sent_id"],
                    text=sent["text"],
                    normalized=sent["normalized"],
                    start=round(start, 3),
                    end=round(end, 3),
                    score=round(matched[si] / total, 3),
                )
            )

        segments.sort(key=lambda s: (s.chapter_id, s.start))
        log.info(
            "align(asr_bridge): %d/%d sentences matched",
            len(segments), len(sentences),
        )
        return segments
