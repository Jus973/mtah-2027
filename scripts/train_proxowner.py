"""make train-proxowner [CFG=configs/rules_v12.yaml]: fit the v12 proximity owner model on the fit
sims (each also with its ball smoothed over proxowner.train_widths frames) -> proxowner.model.

fit_on() is also the cross-fitting entry point for scripts/paired.py. Training always reads the
RAW sims (MTAH_TRAIN_SIM_DIR, default ~/opendata/sim, never the evaluated world), whatever world is being evaluated, and caches
fold models by (games, section) under MTAH_PROX_CACHE. Open data only.
"""

import argparse
import hashlib
import json
import os

import numpy as np

from mtah import config, registry
from mtah.io.atd import load_game
from mtah.learn import proxowner

# The raw sims, NOT SIM_DIR: pointing MTAH_SIM_DIR at a smoothed world must not move training into
# it (that silently happened once, 5 Nov). Same string as an explicit $HOME/opendata/sim, so the
# fold cache keys of pinned runs still hold.
TRAIN_DIR = os.path.expanduser(os.environ.get("MTAH_TRAIN_SIM_DIR", "~/opendata/sim"))
CACHE = os.path.expanduser(os.environ.get("MTAH_PROX_CACHE", "~/opendata/proxowner_cache"))


def rows_for(game, cfg):
    root = os.path.join(TRAIN_DIR, game)
    sp = registry.spec(game, root)
    atd, _ = load_game(game, root)
    gt = np.array([line.split(",")[1].strip() for line in open(os.path.join(root, "gt.csv"))][1:])
    pc = cfg["proxowner"]
    out = []
    for width in pc["train_widths"]:
        views = proxowner.match_views(atd, sp, {}, cfg, width=width)
        out.append(proxowner.training_rows(views, gt, pc["step_frames"]))
    return out, views[2]


def fit_on(games, cfg):
    import lightgbm as lgb
    pc = cfg["proxowner"]
    train_part = {k: pc[k] for k in ("train_widths", "step_frames", "params")}   # not the decode
    if pc.get("turn_features"):
        train_part["turn_features"] = True   # absent when off, so older cache keys still hold
    key = hashlib.sha256(json.dumps([sorted(games), train_part, cfg["ball"], cfg["direction"],
                                     TRAIN_DIR], sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(CACHE, "proxowner-{}.txt".format(key))
    if os.path.isfile(path):
        return lgb.Booster(model_file=path)
    X, y, names = [], [], None
    for g in games:
        parts, names = rows_for(g, cfg)
        for a, b in parts:
            X.append(a); y.append(b)
    booster = proxowner.train(np.vstack(X), np.concatenate(y), pc["params"], names)
    os.makedirs(CACHE, exist_ok=True)
    booster.save_model(path)
    return booster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rules_v12.yaml")
    cfg = config.load(os.path.abspath(parser.parse_args().config))
    prefix = cfg["matches"].get("prefix", "DFL-")
    booster = fit_on([prefix + g for g in cfg["matches"]["fit"]], cfg)
    out = os.path.join(registry.ROOT, cfg["proxowner"]["model"])
    booster.save_model(out)
    print("wrote", os.path.relpath(out, registry.ROOT))


if __name__ == "__main__":
    main()
