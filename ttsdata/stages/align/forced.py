"""Forced aligner (Strategy B) — direct audio<->text forced alignment.

Selectable via ``align.forced.backend``. Same ``align(cfg, chapters, sentences)``
interface as the ASR bridge, so downstream stages and the eval harness are
strategy-agnostic.

Status: NOT yet implemented. These backends need heavy dependencies, model
downloads and real audio, which could not be validated in the build environment,
so they are left as guided stubs rather than shipped untested. Each backend's
shape is documented below for completion.

Backends
--------
torchaudio_mms  (recommended first — pip-installable, CPU-capable, multilingual)
  1. Concatenate per-chapter book text (use the chapter→sentence mapping from a
     first ASR-bridge pass, or align the whole chapter's sentences).
  2. Load ``MMS_FA`` bundle: ``torchaudio.pipelines.MMS_FA`` -> model + tokenizer
     + aligner. Romanise Czech tokens with ``uroman`` as MMS expects.
  3. Run emissions on the 16 kHz ``asr_wav``; ``torchaudio.functional.forced_align``
     (or the bundle's aligner) to get per-token frame spans -> times.
  4. Group token spans back into sentence spans -> Segment(start, end, score)
     where score is the mean token alignment probability.

aeneas  (simplest baseline — sentence-level sync)
  1. Write the chapter's sentences to a plain-text fragment list.
  2. Run aeneas (``aeneas.executetask``) with language='ces', espeak backend,
     to produce a sync map (fragment -> [begin, end]).
  3. Map fragments back to sentences -> Segment. Score is constant (aeneas gives
     no confidence); use 1.0 or a heuristic from fragment duration vs text length.

mfa  (gold-standard boundaries — separate conda env)
  1. Build a per-chapter corpus dir: ``<id>.wav`` + ``<id>.txt`` (sentence text).
  2. Subprocess: ``mfa align <corpus> czech_mfa czech_mfa <out>`` (Czech acoustic
     model + dictionary).
  3. Parse the output TextGrid word tier -> sentence spans -> Segment.
"""

from __future__ import annotations

from ...config import Config
from .base import Segment


class ForcedAligner:
    def align(
        self, cfg: Config, chapters: list[dict], sentences: list[dict]
    ) -> list[Segment]:
        backend = cfg.get("align.forced.backend", "torchaudio_mms")
        raise NotImplementedError(
            f"Forced alignment backend '{backend}' is not implemented yet. "
            "See ttsdata/stages/align/forced.py for the implementation guide."
        )
