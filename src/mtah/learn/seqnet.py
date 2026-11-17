"""Sequence owner model (board #91): a small dilated 1-D convolutional net over the proximity
owner model's per-frame features (with ball-turn features), read at 5 Hz with about +-25 s of
context. It gives a second P(home owns | alive) that is averaged with the LightGBM one before
the 2-state decode in learn/proxowner.py.

Trained with torch on the open DFL sims only (scripts/train_seqnet.py); the weights are exported
to models/seqnet_v1.json (base64 float32 arrays) and inference here is plain numpy (float64), so
the runtime needs no torch.

Architecture (per view): 1x1 conv (2k -> 64) on standardised features plus missing flags, seven
residual blocks x + W2 . gelu(conv3_d(gelu(x))) with dilations 1..64, gelu, 1x1 conv -> logit.
The step-5 sequence is read at each of the 5 offsets so every frame gets a value; seeds are
averaged in probability; both team views are averaged as in proxowner.p_home."""

import base64
import json
import os

import numpy as np
from scipy.special import erf, expit

from mtah import registry

STEP = 5
CLIP = 8.0


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _conv(x, w, b, d=1):
    """Conv1d with zero 'same' padding. x (C_in, T), w (C_out, C_in, K) with K in {1, 3}."""
    if w.shape[2] == 1:
        return w[:, :, 0] @ x + b[:, None]
    T = x.shape[1]
    xp = np.zeros((x.shape[0], T + 2 * d))
    xp[:, d:d + T] = x
    out = b[:, None] + w[:, :, 0] @ xp[:, 0:T] + w[:, :, 1] @ xp[:, d:d + T]
    return out + w[:, :, 2] @ xp[:, 2 * d:2 * d + T]


def read_weights(path):
    """{name: array} from a .json weights file (base64 little-endian arrays with dtype and shape;
    the committed format) or an .npz (the training cache)."""
    if not path.endswith(".json"):
        return dict(np.load(path))
    with open(path) as fh:
        doc = json.load(fh)
    return {k: np.frombuffer(base64.b64decode(v["b64"]), dtype=np.dtype(v["dtype"]).newbyteorder("<"))
            .reshape(v["shape"]) for k, v in doc["arrays"].items()}


def write_weights(arrays, path, meta=None):
    doc = {"format": "mtah-seqnet-1", "meta": meta or {}, "arrays": {}}
    for k in sorted(arrays):
        a = np.asarray(arrays[k])
        if a.dtype.kind == "U":
            doc["meta"][k] = a.tolist()
            continue
        a = a.astype(a.dtype.newbyteorder("<"))
        doc["arrays"][k] = {"dtype": a.dtype.str, "shape": list(a.shape),
                            "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=0, sort_keys=True)
        fh.write("\n")


def feature_names(path):
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)["meta"]["names"]
    return [str(n) for n in np.load(path)["names"]]


class SeqNet:
    def __init__(self, path):
        z = read_weights(path)
        self.mu, self.sd = z["mu"].astype(float), z["sd"].astype(float)
        self.dil = [int(d) for d in z["dil"]]
        self.nets = []
        for s in range(int(z["n_seeds"])):
            g = lambda n: z["s%d_%s" % (s, n)].astype(float)  # noqa: E731
            self.nets.append({
                "in": (g("in_w"), g("in_b")),
                "blocks": [(g("b%d_c1_w" % i), g("b%d_c1_b" % i), g("b%d_c2_w" % i), g("b%d_c2_b" % i))
                           for i in range(len(self.dil))],
                "out": (g("out_w"), g("out_b"))})

    def _prep(self, x):
        miss = ~np.isfinite(x)
        v = np.where(miss, 0.0, (np.nan_to_num(x) - self.mu) / self.sd)
        return np.concatenate([np.clip(v, -CLIP, CLIP), miss], axis=1).T

    def _logit(self, net, x):
        h = _conv(x, *net["in"])
        for (w1, b1, w2, b2), d in zip(net["blocks"], self.dil):
            h = h + _conv(_gelu(_conv(_gelu(h), w1, b1, d)), w2, b2)
        return _conv(_gelu(h), *net["out"])[0]

    def p_view(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[1] != len(self.mu):
            raise ValueError("seqnet expects %d features, got %d" % (len(self.mu), x.shape[1]))
        out = np.zeros(len(x))
        for off in range(STEP):
            xs = self._prep(x[off::STEP])
            out[off::STEP] = np.mean([expit(self._logit(n, xs)) for n in self.nets], axis=0)
        return out

    def p_home(self, views):
        a, b = views[0], views[1]
        return (self.p_view(a) + 1 - self.p_view(b)) / 2


def load(cfg):
    return SeqNet(os.path.join(registry.ROOT, cfg["proxowner"]["seqnet"]["model"]))
