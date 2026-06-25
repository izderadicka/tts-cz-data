"""Czech text normalisation: written form -> spoken form.

The transcript label must match what the narrator actually says, so numbers and
abbreviations are expanded to words ("123" -> "sto dvacet tři", "např." ->
"například"). We keep two views of each sentence:
  * ``text``       — lightly cleaned, punctuation kept (best target for TTS)
  * ``normalized`` — fully spoken-out (used for token alignment vs ASR)

Number expansion uses ``num2words`` (lang 'cs') when available; if it is not
installed the digits are left as-is and a one-time warning is logged.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Abbreviation -> spoken expansion. These are written WITH a trailing period in
# real text, so we require the dot when matching — otherwise the short keys would
# match prefixes of ordinary words (e.g. "kap" inside "Kapitola").
ABBREV_EXPANSIONS = {
    "např": "například",
    "tj": "to jest",
    "tzn": "to znamená",
    "tzv": "takzvaný",
    "atd": "a tak dále",
    "apod": "a podobně",
    "aj": "a jiné",
    "mj": "mimo jiné",
    "resp": "respektive",
    "popř": "popřípadě",
    "kupř": "kupříkladu",
    "č": "číslo",
    "čís": "číslo",
    "str": "strana",
    "roč": "ročník",
    "odst": "odstavec",
    "kap": "kapitola",
    "obr": "obrázek",
    "tab": "tabulka",
    "hod": "hodin",
    "min": "minut",
    "mld": "miliard",
    "mil": "milionů",
    "tis": "tisíc",
}

# Abbreviations commonly written WITHOUT a period; matched on word boundaries.
ABBREV_NODOT = {
    "cca": "cirka",
    "viz": "viz",
}

_NUMBER_RE = re.compile(r"(\d{1,3}(?:[  .]\d{3})+|\d+)(?:,(\d+))?")
_WORD_RE = re.compile(r"[0-9A-Za-zÁ-Žá-ž]+", re.UNICODE)

_num2words = None
_num2words_checked = False


def _get_num2words():
    global _num2words, _num2words_checked
    if not _num2words_checked:
        _num2words_checked = True
        try:
            from num2words import num2words as fn
            _num2words = fn
        except ImportError:
            log.warning(
                "num2words not installed; numbers left as digits. "
                "Install with `pip install num2words`."
            )
    return _num2words


def expand_numbers(text: str) -> str:
    fn = _get_num2words()
    if fn is None:
        return text

    def repl(m: re.Match) -> str:
        int_part = re.sub(r"[  .]", "", m.group(1))
        try:
            words = fn(int(int_part), lang="cs")
        except (ValueError, NotImplementedError):
            return m.group(0)
        if m.group(2):  # decimal part after a comma
            try:
                dec = fn(int(m.group(2)), lang="cs")
                words = f"{words} celá {dec}"
            except (ValueError, NotImplementedError):
                pass
        return words

    return _NUMBER_RE.sub(repl, text)


def _capitalise_like(token: str, expansion: str) -> str:
    """Preserve a leading capital if the abbreviation was capitalised."""
    if token[:1].isupper():
        return expansion[:1].upper() + expansion[1:]
    return expansion


def expand_abbreviations(text: str) -> str:
    # Dotted abbreviations: require the trailing period (and consume it).
    dotted = re.compile(
        r"\b(" + "|".join(sorted(ABBREV_EXPANSIONS, key=len, reverse=True)) + r")\.",
        re.IGNORECASE,
    )
    text = dotted.sub(
        lambda m: _capitalise_like(m.group(1), ABBREV_EXPANSIONS[m.group(1).lower()]),
        text,
    )
    # Bare abbreviations: plain word-boundary match.
    nodot = re.compile(
        r"\b(" + "|".join(sorted(ABBREV_NODOT, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    text = nodot.sub(
        lambda m: _capitalise_like(m.group(1), ABBREV_NODOT[m.group(1).lower()]),
        text,
    )
    return text


def clean_text(text: str) -> str:
    """Light cleanup applied to the human-readable label text."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise a few typographic variants to keep labels consistent.
    text = text.replace("…", "...").replace("–", "-").replace("—", "-")
    return text.strip()


def normalize(text: str) -> str:
    """Full spoken-form normalisation used for alignment tokens."""
    text = clean_text(text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens for sequence alignment against ASR output."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]
