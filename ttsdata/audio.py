"""Thin ffmpeg/ffprobe wrappers for decoding and probing audio.

We shell out to ffmpeg rather than depend on a heavy audio library for the
decode step: it handles opus/mp3/m4a uniformly and is the de-facto standard.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegNotFound(
            f"'{tool}' not found on PATH. Install ffmpeg (e.g. `apt-get install ffmpeg`)."
        )
    return path


def probe(path: str | Path) -> dict:
    """Return ffprobe metadata for the first audio stream + format."""
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    stream = audio_streams[0] if audio_streams else {}
    fmt = info.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0.0) or 0.0),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
    }


def decode_to_wav(
    src: str | Path,
    dst: str | Path,
    sample_rate: int,
    channels: int = 1,
    sample_format: str = "s16",
) -> Path:
    """Decode any input to a PCM WAV at the requested rate/channels.

    ``sample_format`` maps to ffmpeg PCM codecs (s16 -> pcm_s16le).
    """
    ffmpeg = _require("ffmpeg")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = {"s16": "pcm_s16le", "s24": "pcm_s24le", "f32": "pcm_f32le"}.get(
        sample_format, "pcm_s16le"
    )
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-acodec", codec,
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dst


def measure_loudness(path: str | Path) -> dict | None:
    """Measure integrated loudness (EBU R128) via ffmpeg loudnorm analysis.

    Returns the loudnorm JSON block, or None if measurement fails.
    """
    ffmpeg = _require("ffmpeg")
    cmd = [
        ffmpeg, "-i", str(path), "-af",
        "loudnorm=I=-23:TP=-2:LRA=7:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON block to stderr; grab the last {...} object.
    text = proc.stderr
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
