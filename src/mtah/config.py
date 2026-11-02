"""Load a YAML config with single-parent inheritance (`inherit: base.yaml`) and a stable hash."""

import hashlib
import json
import os

import yaml

from mtah.registry import ROOT

CONFIG_DIR = os.path.join(ROOT, "configs")


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(name="base.yaml"):
    # A bare file name always resolves inside configs/: a stray same-named file in the current
    # directory must not silently change the loaded config and its hash. Paths (anything with a
    # separator, or absolute) are taken as given.
    path = name if os.path.isabs(name) or os.sep in name else os.path.join(CONFIG_DIR, name)
    with open(path) as handle:
        cfg = yaml.safe_load(handle) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load(parent), cfg)
    return cfg


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
