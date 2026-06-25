"""Stage 3 — alignment. Dispatches to the configured strategy.

``align.strategy`` selects the aligner; both produce the same Segment manifest
so downstream stages (and the eval harness) are strategy-agnostic.
"""

from __future__ import annotations

from ...config import Config


def get_aligner(cfg: Config):
    strategy = cfg.get("align.strategy", "asr_bridge")
    if strategy == "asr_bridge":
        from .asr_bridge import AsrBridgeAligner
        return AsrBridgeAligner()
    if strategy == "forced":
        from .forced import ForcedAligner
        return ForcedAligner()
    raise ValueError(f"Unknown align.strategy: {strategy!r}")


def run(cfg: Config, book_id: str, force: bool = False):
    raise NotImplementedError(
        "Stage 'align' is not implemented yet (see plan milestone 5)."
    )
