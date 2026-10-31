"""make airlock T=public-check CSV=<candidate.csv>: the public-window bug check, verdict only.

RULES.md: the public 5 minutes check correctness (alignment, team sign, format) and are not a
training or tuning set. This prints pass / fail / broken against configs/base.yaml
`public_check`, the official validator result, and whether swapping H/A would score higher. The
spellF1 itself is never printed, to anyone (mtah.eval.score.public_check).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config  # noqa: E402
from mtah.eval.score import public_check  # noqa: E402
from mtah.submit.write import sha256  # noqa: E402

args = game_args(add=lambda p: p.add_argument("--csv", required=True))
cfg = config.load()["public_check"]
r = public_check(args.csv, cfg)
put("candidate sha256", sha256(args.csv)[:12])
put("verdict", r["verdict"])
ok = check("valid submission (official validator)", r["reason"] is None,
           **({"reason": r["reason"]} if r["reason"] else {}))
if ok:
    ok &= check("not broken: team sign and alignment plausible (spellF1 >= {})".format(
        cfg["broken_below"]), r["verdict"] != "broken")
    ok &= check("team sign: H/A as written beats H/A swapped", not r["swap_scores_higher"])
    ok &= check("public bug check passes (spellF1 >= {}; pass/fail only)".format(cfg["floor"]),
                r["verdict"] == "pass")
sys.exit(0 if ok else 1)
