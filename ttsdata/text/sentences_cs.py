"""Czech-aware sentence segmentation.

Plain-text books wrap lines mid-sentence and contain abbreviations ("tj.",
"např.", "str.") and ordinals ("5. května") whose periods must NOT end a
sentence. This rule-based splitter handles those cases without a heavy NLP
dependency; if ``sentence_splitter`` is installed it can be preferred via config
later. The goal is clean sentence units to anchor against the audio.
"""

from __future__ import annotations

import re

# Uppercase Czech letters (plus common loan letters) — used to detect a plausible
# next-sentence start. Python's ``re`` has no \p{Lu}, so we enumerate.
_UPPER = "A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽÄÖÜ"
_SENT_START = rf'["„«»(\[\d{_UPPER}]'

# Abbreviations whose trailing period does not end a sentence (lowercased, no dot).
ABBREVIATIONS = {
    # general
    "tj", "tzv", "tzn", "atd", "apod", "aj", "mj", "např", "resp", "popř",
    "kupř", "cca", "viz", "č", "čís", "čl", "str", "s", "p", "pí", "sl",
    "roč", "odst", "písm", "zák", "sb", "stol", "tis", "mil", "mld", "obr",
    "tab", "kap", "sv", "nám", "ul", "tř", "fa", "n", "l", "př",
    # time / units
    "hod", "min", "sec", "kč",
    # titles / academia
    "dr", "prof", "doc", "ing", "mgr", "bc", "mudr", "judr", "rndr", "phdr",
    "csc", "drsc", "ph", "d", "rsdr", "paeddr", "thdr",
}

# Match a run of sentence-ending punctuation followed by whitespace.
_BOUNDARY_RE = re.compile(r"([.!?…]+)(\s+)")
# Trailing "word" (letters/digits) immediately before the punctuation.
_PRECEDING_RE = re.compile(rf"([{_UPPER}a-zá-ž0-9]+)$", re.IGNORECASE)


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines; join wrapped lines within a paragraph into spaces."""
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for block in blocks:
        joined = re.sub(r"\s*\n\s*", " ", block).strip()
        joined = re.sub(r"[ \t]+", " ", joined)
        if joined:
            out.append(joined)
    return out


def _is_real_boundary(para: str, punct_end: int, punct: str, after: str) -> bool:
    """Decide whether a punctuation run is a true sentence boundary."""
    # The following sentence must look like a start (capital, quote, digit, ...).
    if not after or not re.match(_SENT_START, after):
        return False
    if "!" in punct or "?" in punct or "…" in punct:
        return True
    # punct is a run of '.'; inspect the token before it.
    before = para[:punct_end - len(punct)]
    m = _PRECEDING_RE.search(before)
    if not m:
        return True
    token = m.group(1)
    low = token.lower()
    if low in ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha():  # initials: "J. Novák"
        return False
    if token.isdigit():
        # Ordinal like "5. května": only a boundary if the next char is uppercase
        # (a digit/quote start after "5." is too ambiguous → keep together).
        return bool(re.match(rf"[{_UPPER}]", after))
    return True


def split_sentences(text: str) -> list[str]:
    """Segment Czech text into a flat list of sentences."""
    sentences: list[str] = []
    for para in _paragraphs(text):
        start = 0
        for m in _BOUNDARY_RE.finditer(para):
            punct_end = m.end(1)
            after_idx = m.end(2)
            after = para[after_idx:after_idx + 1]
            if _is_real_boundary(para, punct_end, m.group(1), after):
                sentences.append(para[start:punct_end].strip())
                start = after_idx
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if s]
