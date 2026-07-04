"""Tests for Czech sentence splitting and normalisation."""

import pytest

from ttsdata.text import normalize_cs, sentences_cs


def test_split_basic():
    text = "Dobrý den. Jak se máte? Mám se dobře!"
    assert sentences_cs.split_sentences(text) == [
        "Dobrý den.",
        "Jak se máte?",
        "Mám se dobře!",
    ]


def test_split_keeps_abbreviation():
    text = "Bydlí na ul. Krátká a má rád pivo. Pak šel domů."
    sents = sentences_cs.split_sentences(text)
    assert sents[0] == "Bydlí na ul. Krátká a má rád pivo."
    assert sents[1] == "Pak šel domů."


def test_split_keeps_ordinal_midsentence():
    text = "Narodil se 5. května v Praze. Bylo to dávno."
    sents = sentences_cs.split_sentences(text)
    assert sents[0] == "Narodil se 5. května v Praze."
    assert len(sents) == 2


def test_split_joins_wrapped_lines():
    text = "Toto je jedna\nvěta rozdělená na řádky. Druhá věta."
    sents = sentences_cs.split_sentences(text)
    assert sents[0] == "Toto je jedna věta rozdělená na řádky."


def test_split_paragraph_boundary():
    text = "První odstavec\n\nDruhý odstavec"
    assert sentences_cs.split_sentences(text) == ["První odstavec", "Druhý odstavec"]


def test_expand_abbreviations():
    out = normalize_cs.expand_abbreviations("Máme např. jablka a tzn. ovoce.")
    assert "například" in out
    assert "to znamená" in out


def test_abbrev_does_not_match_word_prefix():
    # "kap." -> "kapitola" but "Kapitola" must stay untouched (regression).
    assert normalize_cs.expand_abbreviations("Kapitola první") == "Kapitola první"
    assert "kapitola" in normalize_cs.expand_abbreviations("viz kap. 3").lower()


def test_tokenize_lowercases():
    assert normalize_cs.tokenize("Ahoj, Světe!") == ["ahoj", "světe"]


@pytest.mark.skipif(
    normalize_cs._get_num2words() is None, reason="num2words not installed"
)
def test_expand_numbers_cs():
    assert normalize_cs.expand_numbers("123") == "sto dvacet tři"
    assert "celá" in normalize_cs.expand_numbers("3,5")
    # thousands separator
    assert normalize_cs.expand_numbers("1 234").startswith("tisíc")
