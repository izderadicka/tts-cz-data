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
  5. If a sentence's *edge* tokens went unmatched (ASR typically garbles proper
     nouns), the span from step 4 stops short and would truncate the audio
     mid-sentence. The garbled words are still in the ASR stream right past the
     matched span, so the span is extended over them — bounded by the unmatched
     token count, a pause guard, and tokens matched by neighbouring sentences.

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


# A pause longer than this between ASR words is treated as a sentence boundary;
# span extension over garbled edge words never crosses it.
MAX_EXTEND_GAP_S = 0.5


def _extend_forward(
    meta: list[dict], claimed: list[bool], chapter_id: str,
    idx: int, end: float, n_tokens: int,
) -> float:
    """Extend ``end`` over unmatched ASR words following matched index ``idx``.

    Walks at most ``n_tokens + 1`` words (ASR may split one book word into two),
    stopping at a word matched by another sentence, a chapter change, or a pause.
    """
    for j in range(idx + 1, min(idx + n_tokens + 2, len(meta))):
        m = meta[j]
        if claimed[j] or m["chapter_id"] != chapter_id or m["start"] - end > MAX_EXTEND_GAP_S:
            break
        end = max(end, m["end"])
    return end


def _extend_backward(
    meta: list[dict], claimed: list[bool], chapter_id: str,
    idx: int, start: float, n_tokens: int,
) -> float:
    """Mirror of :func:`_extend_forward` for unmatched leading tokens."""
    for j in range(idx - 1, max(idx - n_tokens - 2, -1), -1):
        m = meta[j]
        if claimed[j] or m["chapter_id"] != chapter_id or start - m["end"] > MAX_EXTEND_GAP_S:
            break
        start = min(start, m["start"])
    return start


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
        matched_words = [[] for _ in range(n)]
        asr_ids: list[list[int]] = [[] for _ in range(n)]
        book_pos: list[list[int]] = [[] for _ in range(n)]
        chapters_hit: list[Counter] = [Counter() for _ in range(n)]
        claimed = [False] * len(asr_tokens)

        # First global book-token index per sentence -> token position within it.
        sent_first: dict[int, int] = {}
        for gi, si in enumerate(book_owner):
            sent_first.setdefault(si, gi)

        sm = SequenceMatcher(None, book_tokens, asr_tokens, autojunk=False)
        log.info(f"Start sequence matching - ASR tokens {len(asr_tokens)} vs ebook tokens {len(book_tokens)}")
        for bi, ai, size in sm.get_matching_blocks():
            for k in range(size):
                si = book_owner[bi + k]
                matched[si] += 1
                matched_words[si].append(asr_tokens[ai + k])
                asr_ids[si].append(ai + k)
                book_pos[si].append(bi + k - sent_first[si])
                chapters_hit[si][asr_meta[ai + k]["chapter_id"]] += 1
                claimed[ai + k] = True
        log.info("Matching done")
        segments: list[Segment] = []
        for si, sent in enumerate(sentences):
            total = len(sent["tokens"])
            if matched[si] == 0 or total == 0:
                continue
            start = min(asr_meta[i]["start"] for i in asr_ids[si])
            end = max(max(asr_meta[i]["end"] for i in asr_ids[si]), start)
            chapter_id = chapters_hit[si].most_common(1)[0][0]

            # Recover audio of garbled (unmatched) edge words — see module doc, step 5.
            lead = min(book_pos[si])
            trail = total - 1 - max(book_pos[si])
            if trail > 0:
                end = _extend_forward(
                    asr_meta, claimed, chapter_id, max(asr_ids[si]), end, trail
                )
            if lead > 0:
                start = _extend_backward(
                    asr_meta, claimed, chapter_id, min(asr_ids[si]), start, lead
                )
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
                    matched_words=matched_words[si],
                )
            )

        segments.sort(key=lambda s: (s.chapter_id, s.start))
        log.info(
            "align(asr_bridge): %d/%d sentences matched",
            len(segments), len(sentences),
        )
        return segments
