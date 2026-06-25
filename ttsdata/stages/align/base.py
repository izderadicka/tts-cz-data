"""Common alignment interface shared by both strategies.

An aligner maps a chapter's audio + the book's clean sentences to a list of
``Segment`` objects (sentence text anchored to a [start, end] time span with a
confidence score). Strategies (``asr_bridge``, ``forced``) are interchangeable
behind this interface so they can be compared on identical inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from ...config import Config


@dataclass
class Segment:
    book_id: str
    chapter_id: str
    seg_id: str
    text: str          # clean book sentence (with punctuation) — the TTS label
    normalized: str    # fully spoken-out form
    start: float       # seconds
    end: float         # seconds
    score: float       # alignment confidence in [0, 1]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


class Aligner(Protocol):
    """Strategy interface. Implementations live in sibling modules."""

    def align_chapter(
        self, cfg: Config, chapter: dict, sentences: list[dict]
    ) -> list[Segment]:
        """Return ordered segments for one chapter.

        ``chapter`` is an ingest/transcribe record (carries ``asr_wav`` and,
        for the ASR bridge, time-stamped ``words``). ``sentences`` are the
        book's stage-1 sentence records for this book.
        """
        ...
