"""Command-line orchestrator for the ttsdata pipeline.

Usage examples:
    ttsdata list-books
    ttsdata run --stage ingest --book mybook
    ttsdata run --book mybook                 # all stages, in order
    ttsdata run                               # all books, all stages
    ttsdata run --config config/cuda.yaml ... # override defaults
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from .config import load_config
from .workspace import list_books

# Ordered pipeline stages and the module that implements each.
STAGE_ORDER = [
    ("ingest", "ttsdata.stages.ingest"),
    ("text_prep", "ttsdata.stages.text_prep"),
    ("transcribe", "ttsdata.stages.transcribe"),
    ("align", "ttsdata.stages.align"),
    ("segment", "ttsdata.stages.segment"),
    ("quality", "ttsdata.stages.quality"),
    ("export", "ttsdata.stages.export"),
]
STAGE_NAMES = [name for name, _ in STAGE_ORDER]
_STAGE_MODULE = dict(STAGE_ORDER)


def _run_stage(stage: str, cfg, book_id: str, force: bool) -> None:
    module = importlib.import_module(_STAGE_MODULE[stage])
    logging.info("=== stage %s — book %s ===", stage, book_id)
    module.run(cfg, book_id, force=force)


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    books = [args.book] if args.book else list_books(cfg)
    if not books:
        logging.error("No books found under %s", cfg.path("paths.raw"))
        return 1

    if args.stage:
        stages = [args.stage]
    else:
        start = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0
        stages = STAGE_NAMES[start:]

    for book_id in books:
        for stage in stages:
            _run_stage(stage, cfg, book_id, args.force)
    return 0


def cmd_list_books(args) -> int:
    cfg = load_config(args.config)
    books = list_books(cfg)
    if not books:
        print(f"(no books under {cfg.path('paths.raw')})")
        return 0
    for book in books:
        print(book)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttsdata", description=__doc__)
    parser.add_argument("--config", help="Override config YAML merged on defaults")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run pipeline stage(s)")
    p_run.add_argument("--book", help="Book id (default: all books)")
    p_run.add_argument(
        "--stage", choices=STAGE_NAMES, help="Run only this stage"
    )
    p_run.add_argument(
        "--from-stage", dest="from_stage", choices=STAGE_NAMES,
        help="Run from this stage to the end",
    )
    p_run.add_argument(
        "--force", action="store_true", help="Redo even if outputs exist"
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list-books", help="List discovered books")
    p_list.set_defaults(func=cmd_list_books)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
