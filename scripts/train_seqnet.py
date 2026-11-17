"""Train the sequence owner model (board #91, learn/seqnet.py) on the open DFL fit sims.

  python scripts/train_seqnet.py --config configs/rules_v12_hedgeA_seq.yaml
      -> writes cfg.proxowner.seqnet.model (.json, base64 float32 weights; inference needs no torch)

Training needs torch (CPU; a dev-time dependency only, not in requirements.txt). Like
train_proxowner.py it always reads the RAW fit sims (MTAH_TRAIN_SIM_DIR), at the proxowner
train widths, both team views, 5 Hz offsets 0 and 2; fitted models are cached by content key so
the paired cross-fit (scripts/paired.py) reuses them. Never touches the match bundle.

  --seed-cache DIR  export board #91 experiment checkpoints (cnn_<fold>_s<seed>.pt, trained by
                    this same code in scripts/exp91.py) into the cache instead of retraining.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mtah import config, registry  # noqa: E402
from train_proxowner import TRAIN_DIR  # noqa: E402

CACHE = os.path.expanduser("~/opendata/seqnet_cache")
STEP = 5
DIL = (1, 2, 4, 8, 16, 32, 64)
CH = 64
TRAIN_KEYS = ("steps", "batch", "crop", "seeds", "lr", "weight_decay", "dropout")


def train_part(cfg):
    sq = cfg["proxowner"]["seqnet"]
    return {"widths": cfg["proxowner"]["train_widths"], **{k: sq[k] for k in TRAIN_KEYS}}


def cache_path(games, cfg):
    key = hashlib.sha256(json.dumps([sorted(games), train_part(cfg), cfg["ball"], cfg["direction"],
                                     TRAIN_DIR], sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(CACHE, "seqnet-{}.npz".format(key))


def build_net(k_in, p_drop):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = nn.Conv1d(CH, CH, 3, dilation=d, padding=d)
            self.c2 = nn.Conv1d(CH, CH, 1)
            self.act = nn.GELU()
            self.drop = nn.Dropout(p_drop)

        def forward(self, x):
            return x + self.c2(self.drop(self.act(self.c1(self.act(x)))))

    return nn.Sequential(nn.Conv1d(2 * k_in, CH, 1), *[Block(d) for d in DIL],
                         nn.GELU(), nn.Conv1d(CH, 1, 1))


def prep(x, mu, sd):
    miss = ~np.isfinite(x)
    v = np.where(miss, 0.0, (np.nan_to_num(x) - mu) / sd)
    return np.concatenate([np.clip(v, -8, 8), miss], axis=1).T.astype(np.float32)


def sequences(game, cfg):
    """[(x (T5,k), y (T5,), w (T5,))] for one RAW fit sim: every train width, both views,
    offsets 0 and 2. Features always include the ball-turn columns."""
    from mtah.io.atd import load_game
    from mtah.learn import proxowner
    root = os.path.join(TRAIN_DIR, game)
    sp = registry.spec(game, root)
    atd, _ = load_game(game, root)
    gt = np.array([ln.split(",")[1].strip() for ln in open(os.path.join(root, "gt.csv"))][1:])
    out, names = [], None
    for width in cfg["proxowner"]["train_widths"]:
        a, b, names = proxowner.match_views(atd, sp, {}, cfg, width=width, turns=True)
        for view, lab in ((a, "H"), (b, "A")):
            for off in (0, 2):
                out.append((np.ascontiguousarray(view[off::STEP], np.float32),
                            (gt[off::STEP] == lab).astype(np.float32),
                            np.isin(gt[off::STEP], ["H", "A"]).astype(np.float32)))
    return out, names


def standardiser(seqs):
    allx = np.concatenate([x[w > 0] for x, _, w in seqs])
    mu, sd = np.nanmean(allx, axis=0), np.nanstd(allx, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    return mu, sd


def train_one(seqs, mu, sd, seed, sq, threads):
    import torch
    torch.set_num_threads(threads)
    X = [torch.from_numpy(prep(x, mu, sd)) for x, _, _ in seqs]
    Y = [torch.from_numpy(y) for _, y, _ in seqs]
    W = [torch.from_numpy(w) for _, _, w in seqs]
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = build_net(len(mu), sq["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=sq["lr"], weight_decay=sq["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=sq["lr"], total_steps=sq["steps"])
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    net.train()
    for it in range(sq["steps"]):
        xb, yb, wb = [], [], []
        for _ in range(sq["batch"]):
            i = rng.randint(len(X))
            s = rng.randint(0, max(1, X[i].shape[1] - sq["crop"]))
            xb.append(X[i][:, s:s + sq["crop"]]); yb.append(Y[i][s:s + sq["crop"]])
            wb.append(W[i][s:s + sq["crop"]])
        L = min(v.shape[1] for v in xb)
        xb = torch.stack([v[:, :L] for v in xb]); yb = torch.stack([v[:L] for v in yb])
        wb = torch.stack([v[:L] for v in wb])
        loss = (lossf(net(xb)[:, 0], yb) * wb).sum() / wb.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            print("seed", seed, "step", it, "loss", round(float(loss), 4), flush=True)
    net.eval()
    return net.state_dict()


def export(states, mu, sd, names, path):
    """torch state dicts (one per seed) -> the numpy .npz learn/seqnet.py reads."""
    arr = {"mu": np.asarray(mu, float), "sd": np.asarray(sd, float), "dil": np.array(DIL),
           "n_seeds": np.array(len(states)), "names": np.array(names)}
    for s, st in enumerate(states):
        g = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in st.items()}
        arr["s%d_in_w" % s], arr["s%d_in_b" % s] = g["0.weight"], g["0.bias"]
        for i in range(len(DIL)):
            for c in ("c1", "c2"):
                arr["s%d_b%d_%s_w" % (s, i, c)] = g["%d.%s.weight" % (i + 1, c)]
                arr["s%d_b%d_%s_b" % (s, i, c)] = g["%d.%s.bias" % (i + 1, c)]
        n = len(DIL) + 2
        arr["s%d_out_w" % s], arr["s%d_out_b" % s] = g["%d.weight" % n], g["%d.bias" % n]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, path)


def fit_on(games, cfg, threads=6):
    """Path of the fitted .npz for these RAW fit sims (cached by content key)."""
    path = cache_path(games, cfg)
    if os.path.isfile(path):
        return path
    import torch  # noqa: F401 -- before lightgbm-linked code paths load their OpenMP runtime
    sq = cfg["proxowner"]["seqnet"]
    seqs, names = [], None
    for g in games:
        s, names = sequences(g, cfg)
        seqs += s
    mu, sd = standardiser(seqs)
    states = [train_one(seqs, mu, sd, seed, sq, threads) for seed in sq["seeds"]]
    export(states, mu, sd, names, path)
    return path


def seed_cache(cfg, games, src):
    """Export exp91 checkpoints (same training code) into the cache: folds named by the left-out
    game, 'ALL' for all five."""
    import torch
    _, names = sequences(games[0], cfg)
    folds = [(g, [h for h in games if h != g]) for g in games] + [("ALL", games)]
    for key, fit in folds:
        k = key.replace("DFL-", "")
        cks = [torch.load(os.path.join(src, "cnn_{}_s{}.pt".format(k, s)), weights_only=False)
               for s in cfg["proxowner"]["seqnet"]["seeds"]]
        path = cache_path(fit, cfg)
        export([c["state"] for c in cks], cks[0]["mu"], cks[0]["sd"], names, path)
        print("seeded", key, "->", os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rules_v12_hedgeA_seq.yaml")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed-cache", default="")
    a = ap.parse_args()
    cfg = config.load(os.path.abspath(a.config))
    prefix = cfg["matches"].get("prefix", "DFL-")
    games = [prefix + g for g in cfg["matches"]["fit"]]
    if a.seed_cache:
        seed_cache(cfg, games, os.path.expanduser(a.seed_cache))
    src = fit_on(games, cfg, a.threads)
    from mtah.learn import seqnet
    out = os.path.join(registry.ROOT, cfg["proxowner"]["seqnet"]["model"])
    seqnet.write_weights(dict(np.load(src)), out)
    print("wrote", os.path.relpath(out, registry.ROOT))


if __name__ == "__main__":
    main()
