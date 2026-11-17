"""Data-free determinism check for the board #91 sequence owner model: generate the fake match,
run the pipeline with CFG and print sha256 of the labels and of the seqnet probabilities.
Compare the printed hashes across hosts / Docker platforms (open data only, no bundle)."""
import hashlib
import os
import subprocess
import sys

import numpy as np

from mtah import config, pipeline, registry
from mtah.learn import proxowner, seqnet

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/rules_v12_hedgeA_seq.yaml"
root = os.path.join(registry.ROOT, "fakedata", "out", "BAR-FAKE")
if not os.path.isdir(root):
    subprocess.check_call([sys.executable, os.path.join(registry.ROOT, "fakedata", "generate.py"),
                           "--game", "BAR-FAKE", "--seed", "0", "--half-s", "900"],
                          stdout=subprocess.DEVNULL)
cfg = config.load(os.path.join(registry.ROOT, cfg_path))
labels, info = pipeline.run("BAR-FAKE", cfg, root)
print("labels", hashlib.sha256("".join(labels).encode()).hexdigest()[:16])
from mtah.io.atd import load_game  # noqa: E402
atd, _ = load_game("BAR-FAKE", root)
sp = registry.spec("BAR-FAKE", root)
v = proxowner.match_views(atd, sp, {}, cfg, turns=True)
p = seqnet.load(cfg).p_home(v)
print("p_seq", hashlib.sha256(np.round(p, 9).tobytes()).hexdigest()[:16],
      "p_seq_raw", hashlib.sha256(p.tobytes()).hexdigest()[:16])
