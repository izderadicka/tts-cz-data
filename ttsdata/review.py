"""Human review verdicts: persist approve/reject decisions and fold them in.

``ttsdata listen`` records one verdict per clip in
``<work>/<book>/review/verdicts.csv`` (``clip_id,verdict,flags``; the flags
column is audition-time context only). :func:`apply_verdicts` merges verdicts
into a quality manifest: it is a pure, idempotent function of the current
clips × verdicts, so it can run both from ``ttsdata review-apply`` and at the
end of every quality run — verdicts survive ``--force`` re-runs.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

VERDICTS = ("approve", "reject")


def load_verdicts(path: Path) -> dict[str, str]:
    """Read verdicts.csv into {clip_id: verdict}; {} if the file is missing."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r["verdict"] for r in csv.DictReader(fh)}


def save_verdict(path: Path, clip_id: str, verdict: str, flags: str = "") -> None:
    """Record (or overwrite) one clip's verdict, rewriting the whole file.

    The file stays small and sorted by clip_id, and every keypress in the
    listen tool is durable on its own.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict: {verdict!r} (expected approve | reject)")
    rows: dict[str, tuple[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            rows = {r["clip_id"]: (r["verdict"], r.get("flags", "")) for r in csv.DictReader(fh)}
    rows[clip_id] = (verdict, flags)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "verdict", "flags"])
        for cid in sorted(rows):
            writer.writerow([cid, rows[cid][0], rows[cid][1]])


def apply_verdicts(clips: list[dict], verdicts: dict[str, str]) -> dict[str, int]:
    """Fold verdicts into clip records in place; returns outcome counts.

    * ``reject`` — a human said it's bad: status becomes ``reject`` whatever
      the checks decided.
    * ``approve`` — lifts a soft-flag ``review`` to ``pass``. Hard-fail
      ``reject`` statuses are objective (duration, CER) and are not lifted.

    Flags are never modified, so the manifest keeps full provenance
    (``flags: ["edge_noise"], verdict: "approve", status: "pass"``).
    """
    counts = {"approved": 0, "rejected": 0, "stale": 0}
    by_id = {c["clip_id"]: c for c in clips}
    for clip_id, verdict in verdicts.items():
        clip = by_id.get(clip_id)
        if clip is None:
            log.warning("review: verdict for unknown clip %s (stale after re-segmentation?)", clip_id)
            counts["stale"] += 1
            continue
        clip["verdict"] = verdict
        if verdict == "reject":
            clip["status"] = "reject"
            counts["rejected"] += 1
        elif clip["status"] == "review":
            clip["status"] = "pass"
            counts["approved"] += 1
        elif clip["status"] == "reject":
            log.warning("review: approve cannot lift hard-rejected clip %s", clip_id)
    if any(counts.values()):
        log.info("review: applied verdicts — approved=%d rejected=%d stale=%d",
                 counts["approved"], counts["rejected"], counts["stale"])
    return counts
