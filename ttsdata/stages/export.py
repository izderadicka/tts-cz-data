"""Stage 6 — dataset assembly & export.

Collects clips with ``status == pass`` into a training-ready layout:

    dataset/<book_id>/
      wavs/<clip_id>.wav
      metadata.csv            # pipe-separated, per export.format (below)
      metadata_train.csv / metadata_val.csv
      dataset_stats.json
      PIPER.md                # ready-to-run preprocessing notes

``export.format`` selects the metadata row shape:

  * ``ljspeech`` — ``clip_id|text|normalized`` (universal; Piper, Coqui/VITS,
    Tacotron all consume it)
  * ``piper``    — ``clip_id.wav|normalized`` (Piper's two-column form: wav
    file name + the text to phonemize, nothing to strip at train time)

Audio is resampled to ``export.sample_rate`` if needed. The val split is a
deterministic fraction so runs are reproducible.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..config import Config
from ..manifest import read_jsonl, write_json
from ..workspace import stage_manifest

log = logging.getLogger(__name__)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    # Linear resample — adequate for export; a polyphase resampler can replace it.
    duration = len(audio) / sr_in
    n_out = int(round(duration * sr_out))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _stats(records: list[dict]) -> dict:
    durs = [r["duration"] for r in records]
    total = sum(durs)
    speakers = sorted({r.get("speaker_id") for r in records})
    chars = sorted({ch for r in records for ch in r["text"]})
    return {
        "clips": len(records),
        "total_hours": round(total / 3600, 3),
        "duration_min": round(min(durs), 2) if durs else 0,
        "duration_max": round(max(durs), 2) if durs else 0,
        "duration_mean": round(total / len(durs), 2) if durs else 0,
        "speakers": speakers,
        "char_inventory": "".join(chars),
        "char_count": len(chars),
    }


def run(cfg: Config, book_id: str, force: bool = False) -> dict:
    dataset_dir = cfg.path("paths.dataset") / book_id
    metadata_path = dataset_dir / "metadata.csv"
    if metadata_path.exists() and not force:
        log.info("export: %s already done (use --force to redo)", book_id)
        return {}

    clips = read_jsonl(stage_manifest(cfg, book_id, "quality"))
    if not clips:
        raise FileNotFoundError(f"No quality manifest for {book_id}; run quality first.")

    kept = [c for c in clips if c.get("status") == "pass"]
    if not kept:
        log.warning("export: %s — no clips passed quality; nothing to export", book_id)
        return {}

    fmt = cfg.get("export.format", "ljspeech")
    if fmt not in ("ljspeech", "piper"):
        raise ValueError(f"Unknown export.format: {fmt!r} (expected ljspeech | piper)")
    out_sr = int(cfg.get("export.sample_rate", 22050))
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for c in tqdm(kept, desc=f"export: {book_id}"):
        audio, sr = sf.read(c["wav"], dtype="float32", always_2d=False)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
        audio = _resample(audio, sr, out_sr)
        dst = wav_dir / f"{c['clip_id']}.wav"
        sf.write(dst, audio, out_sr, subtype="PCM_16")
        rows.append(c)

    # Deterministic train/val split.
    rng = random.Random(int(cfg.get("export.seed", 42)))
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * float(cfg.get("export.val_fraction", 0.02))))
    val_ids = {r["clip_id"] for r in shuffled[:n_val]}

    def _write_metadata(path, records):
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                norm = r["normalized"].replace("|", " ")
                if fmt == "piper":
                    fh.write(f"{r['clip_id']}.wav|{norm}\n")
                else:
                    text = r["text"].replace("|", " ")
                    fh.write(f"{r['clip_id']}|{text}|{norm}\n")

    _write_metadata(metadata_path, rows)
    _write_metadata(dataset_dir / "metadata_train.csv", [r for r in rows if r["clip_id"] not in val_ids])
    _write_metadata(dataset_dir / "metadata_val.csv", [r for r in rows if r["clip_id"] in val_ids])

    stats = _stats(rows)
    stats["val_clips"] = len(val_ids)
    stats["sample_rate"] = out_sr
    write_json(dataset_dir / "dataset_stats.json", stats)
    _write_piper_notes(cfg, dataset_dir, out_sr, fmt)

    log.info("export: %s — %d clips, %.2f h -> %s",
             book_id, stats["clips"], stats["total_hours"], dataset_dir)
    return stats


def _write_piper_notes(cfg: Config, dataset_dir, out_sr: int, fmt: str) -> None:
    lang = cfg.get("language", "cs")
    if fmt == "piper":
        layout = "Piper's two-column form (`metadata.csv`: `file.wav|normalized`)"
    else:
        layout = "LJSpeech format (`metadata.csv`: `id|text|normalized`)"
    notes = f"""# Training this dataset with Piper

This dataset is in {layout}.

## Preprocess

```bash
python -m piper_train.preprocess \\
  --language {lang} \\
  --input-dir {dataset_dir} \\
  --output-dir {dataset_dir}/piper \\
  --dataset-format ljspeech \\
  --single-speaker \\
  --sample-rate {out_sr}
```

espeak-ng provides Czech ('{lang}') phonemes. After preprocessing, run
`python -m piper_train ...` per the Piper docs. The same `metadata.csv` also
works directly with Coqui TTS / VITS recipes.
"""
    (dataset_dir / "PIPER.md").write_text(notes, encoding="utf-8")
