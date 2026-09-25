"""YAML configuration with inheritance, dotted CLI overrides and attribute access."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Config(dict):
    """A dict whose keys are also readable as attributes (``cfg.gbdt.params``)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted: str, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent:
        base = _load_raw((path.parent / parent).resolve())
        raw = _merge(base, raw)
    return raw


def _set_dotted(tree: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | tuple = ()) -> Config:
    """Load ``path`` (following ``inherit:``), apply ``key.sub=value`` overrides.

    Relative entries under ``paths`` are resolved against the package root
    (the parent of the ``configs/`` directory), so commands work from any CWD.
    """
    path = Path(path).resolve()
    raw = _load_raw(path)
    for item in overrides:
        key, _, val = item.partition("=")
        _set_dotted(raw, key.strip(), yaml.safe_load(val))
    cfg = _wrap(raw)
    root = path.parent.parent
    for key, val in list(cfg.get("paths", {}).items()):
        if isinstance(val, str):
            p = Path(val)
            cfg.paths[key] = str(p if p.is_absolute() else (root / p).resolve())
    cfg.paths["root"] = str(root)
    return cfg
