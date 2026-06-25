"""Stage 3 — alignment. Dispatches to the configured strategy.

``align.strategy`` selects the aligner; both produce the same Segment manifest
so downstream stages (and the eval harness) are strategy-agnostic.
"""

from __future__ import annotations

import logging

from ...config import Config
from ...manifest import read_jsonl, write_jsonl
from ...workspace import stage_manifest

log = logging.getLogger(__name__)


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
    out_path = stage_manifest(cfg, book_id, "align")
    if out_path.exists() and not force:
        log.info("align: %s already done (use --force to redo)", book_id)
        return read_jsonl(out_path)

    chapters = read_jsonl(stage_manifest(cfg, book_id, "transcribe"))
    sentences = read_jsonl(stage_manifest(cfg, book_id, "text_prep"))
    if not chapters or not sentences:
        raise FileNotFoundError(
            f"align needs transcribe + text_prep manifests for {book_id}"
        )

    aligner = get_aligner(cfg)
    segments = aligner.align(cfg, chapters, sentences)

    min_score = float(cfg.get("align.min_score", 0.5))
    kept = [s.to_dict() for s in segments if s.score >= min_score]
    write_jsonl(out_path, kept)
    log.info(
        "align(%s): %d segments kept (score>=%.2f) of %d",
        cfg.get("align.strategy"), len(kept), min_score, len(segments),
    )
    return kept
