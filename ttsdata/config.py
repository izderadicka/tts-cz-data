"""Configuration loading with dotted-key access and override merging.

The pipeline is config-driven: ``config/default.yaml`` holds every knob, and an
optional override file is deep-merged on top. Access values with dotted keys so
call sites read naturally, e.g. ``cfg.get("asr.model")``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Resolve the bundled default config relative to the repo root (two levels up).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Read-only view over the merged config tree with dotted-key access."""

    def __init__(self, data: dict, repo_root: Path = _REPO_ROOT):
        self._data = data
        self.repo_root = repo_root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {dotted_key}")
        return value

    def path(self, dotted_key: str) -> Path:
        """Resolve a configured path relative to the repo root if not absolute."""
        raw = Path(self.require(dotted_key))
        return raw if raw.is_absolute() else self.repo_root / raw

    @property
    def data(self) -> dict:
        return copy.deepcopy(self._data)


def load_config(override_path: str | Path | None = None) -> Config:
    """Load the default config, deep-merging an optional override file on top."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        data = _deep_merge(data, override)
    return Config(data)
