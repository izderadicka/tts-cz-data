"""Interactive clip auditioning: play segmented clips and show their text.

Reads a segment/quality ``clips.jsonl``, the review ``flagged.csv``, or an
exported ``metadata.csv`` (pipe-separated, wavs in the sibling ``wavs/``) and plays
each clip's WAV via ffplay (part of the ffmpeg system dependency), printing the
label text. Advance with any key; ``r`` replays, ``b`` goes back, ``q`` quits.
A keypress during playback stops the clip and acts immediately.
"""

from __future__ import annotations

import csv
import random
import subprocess
import sys
from pathlib import Path

from .audio import _require
from .manifest import read_jsonl


def _load_clips(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            if "|" in fh.readline():
                return _load_metadata(path)
            fh.seek(0)
            return list(csv.DictReader(fh))
    return read_jsonl(path)


def _load_metadata(path: Path) -> list[dict]:
    """Exported metadata.csv: pipe-separated, headerless.

    Handles both row shapes (``file.wav|text`` and ``clip_id|text|normalized``);
    wav files live in the ``wavs/`` dir next to the metadata file.
    """
    wav_dir = path.parent / "wavs"
    clips = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("|")
            if not fields[0]:
                continue
            clip_id = fields[0].removesuffix(".wav")
            clips.append({
                "clip_id": clip_id,
                "text": fields[1] if len(fields) > 1 else "",
                "wav": str(wav_dir / f"{clip_id}.wav"),
            })
    return clips


def _read_key() -> str:
    """Block for one keypress (cbreak mode); line-based fallback off a TTY."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "q"  # EOF
        return (line.strip() or "n")[0].lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def _show(clip: dict, idx: int, total: int) -> None:
    parts = [f"[{idx + 1}/{total}] {clip.get('clip_id', clip.get('wav'))}"]
    if clip.get("duration"):
        parts.append(f"({clip['duration']}s)")
    if clip.get("status"):
        parts.append(clip["status"])
    if clip.get("flags"):
        flags = clip["flags"]
        parts.append("|".join(flags) if isinstance(flags, list) else flags)
    if clip.get("cer") not in (None, ""):
        parts.append(f"cer={clip['cer']}")
    print("\n" + " ".join(str(p) for p in parts))
    print(f"  text: {clip.get('text', '')}")
    if clip.get("asr_text"):
        print(f"  asr:  {clip['asr_text']}")


def _play(ffplay: str, wav: str) -> subprocess.Popen | None:
    if not Path(wav).exists():
        print(f"  !! wav not found: {wav}")
        return None
    return subprocess.Popen(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def main(args) -> int:
    path = Path(args.manifest)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1
    clips = _load_clips(path)
    if args.status:
        clips = [c for c in clips if c.get("status") == args.status]
    if not clips:
        print("No clips to play.", file=sys.stderr)
        return 1
    if args.random:
        random.shuffle(clips)

    ffplay = _require("ffplay")
    print(f"{len(clips)} clips — any key: next, r: replay, b: back, q: quit")

    idx = 0
    while 0 <= idx < len(clips):
        clip = clips[idx]
        _show(clip, idx, len(clips))
        proc = _play(ffplay, clip.get("wav", ""))
        try:
            key = _read_key()
        finally:
            if proc is not None:
                proc.terminate()
        if key == "q":
            break
        elif key == "r":
            continue
        elif key == "b":
            idx = max(0, idx - 1)
        else:
            idx += 1
    print()
    return 0
