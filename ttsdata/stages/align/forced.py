"""Forced aligner (Strategy B) — direct audio<->text forced alignment.

Backends (config ``align.forced.backend``):
  * ``aeneas``          — sentence-level sync, espeak 'cs' (simplest baseline)
  * ``torchaudio_mms``  — CTC forced alignment with the multilingual MMS model
                          (pip-installable, CPU-capable, supports Czech)
  * ``mfa``             — Montreal Forced Aligner (gold-standard boundaries; runs
                          in its own conda env, invoked as a subprocess)

Implemented in plan milestone 8. The class is wired now so the strategy is
selectable and the eval harness can compare it against the ASR bridge.
"""

from __future__ import annotations

from ...config import Config
from .base import Segment


class ForcedAligner:
    def align(
        self, cfg: Config, chapters: list[dict], sentences: list[dict]
    ) -> list[Segment]:
        backend = cfg.get("align.forced.backend", "aeneas")
        raise NotImplementedError(
            f"Forced alignment backend '{backend}' not implemented yet "
            "(plan milestone 8)."
        )
