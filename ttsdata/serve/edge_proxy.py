"""Stopgap Edge TTS proxy — Czech "Read Aloud" voices over local HTTP.

A temporary bridge that lets a browser reader (foliate-js) speak with Microsoft
Edge's Czech neural voices (Antonín, Vlasta) *while a proper Piper Czech voice is
being trained*. It is deliberately **not** part of the dataset pipeline and lives
in the optional ``serve`` extra so the core install stays untouched.

Why a server-side proxy at all: browser JavaScript cannot set the WebSocket
handshake headers Edge's endpoint expects, so the page POSTs text here and gets
MP3 audio back. The heavy lifting is the unofficial ``edge-tts`` library, which
talks to an **unofficial Microsoft endpoint that may break without notice** —
hence "stopgap". Delete this module once the Piper voice ships.

Endpoints:
  * ``POST /tts``    — JSON in, streamed ``audio/mpeg`` out
  * ``GET  /voices`` — the Czech (``cs``) voices Edge offers

Run it with::

    python -m ttsdata.serve.edge_proxy          # 127.0.0.1:8899
    ttsdata-edge-proxy --host 0.0.0.0 --port 9000

Host/port also come from ``EDGE_PROXY_HOST`` / ``EDGE_PROXY_PORT`` (a CLI flag
wins over the env var).
"""

from __future__ import annotations

import argparse
import logging
import os
from xml.sax.saxutils import escape

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_VOICE = "cs-CZ-AntoninNeural"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8899

app = FastAPI(
    title="Edge TTS proxy (stopgap)",
    description="Czech Edge neural voices for a browser reader, pending Piper.",
)

# CORS: wide open so any local reader origin can call us during development.
# In production, replace ``allow_origins=["*"]`` with the reader's real origin,
# e.g. ``allow_origins=["https://reader.example.org"]`` (a list of exact
# origins; "*" cannot be combined with credentialed requests).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    """Body of a ``POST /tts`` call.

    ``rate``/``pitch`` are Edge's prosody forms — ``"+10%"``, ``"-5%"``,
    ``"+0Hz"`` — and are forwarded to ``Communicate`` only when supplied.
    """

    text: str
    voice: str = DEFAULT_VOICE
    rate: str | None = Field(default=None)
    pitch: str | None = Field(default=None)


@app.post("/tts")
async def tts(req: TTSRequest) -> StreamingResponse:
    """Synthesize ``req.text`` and stream the MP3 back as ``audio/mpeg``."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Edge wraps our text in SSML server-side, so escape XML metacharacters
    # (&, <, >) to keep the input inert and the request well-formed.
    safe_text = escape(text)

    # Only pass rate/pitch through when given — Communicate rejects empty forms.
    kwargs: dict[str, str] = {"voice": req.voice}
    if req.rate:
        kwargs["rate"] = req.rate
    if req.pitch:
        kwargs["pitch"] = req.pitch

    communicate = edge_tts.Communicate(safe_text, **kwargs)

    async def audio_stream():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


@app.get("/voices")
async def voices() -> list[dict]:
    """Return only the Czech (``Locale`` starting ``cs``) Edge voices."""
    all_voices = await edge_tts.list_voices()
    return [v for v in all_voices if v.get("Locale", "").startswith("cs")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttsdata-edge-proxy", description=__doc__.split("\n", 1)[0]
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("EDGE_PROXY_HOST", DEFAULT_HOST),
        help=f"Bind host (default: {DEFAULT_HOST}, or $EDGE_PROXY_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EDGE_PROXY_PORT", DEFAULT_PORT)),
        help=f"Bind port (default: {DEFAULT_PORT}, or $EDGE_PROXY_PORT)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    log.info("Edge TTS proxy (stopgap) on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
