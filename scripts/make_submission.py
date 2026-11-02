"""make submission [TEAM="<name>"] (agents: make airlock T=submission [NAME="<name>"]).

Runs the final candidate (configs/final.yaml, which inherits the selected config) on the match and
writes submissions/final/<team>.csv (team: configs/base.yaml unless given): gitignored,
bundle-derived, never read by an agent. Then:
- the official validator on that exact file;
- the structural bands (PLAN.md §6.3), so the final run fails loudly if anything drifted;
- its sha256, written to submissions/final/<team>.sha256 (shasum format, committed);
- whether it equals the committed sha256 (the M4 reproducibility check: clean clone, twice).
Prints aggregates and the hash only. Deterministic: CPU, single-threaded LightGBM, no randomness.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.eval.bands import bands  # noqa: E402
from mtah.submit.write import sha256, validate, write_csv  # noqa: E402

FINAL = os.path.join(registry.ROOT, "submissions", "final")


def add(parser):
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--name", default=None, help="team name = file name (default: configs team)")


args = game_args(add=add)
args.name = args.name or config.load()["team"]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,59}", args.name):
    sys.exit("--name: the team name; letters, digits, spaces, _ and - only")
cfg = config.load(os.path.abspath(args.config))
sp = registry.spec(args.game, args.data_dir)
labels, info = pipeline.run(args.game, cfg, args.data_dir)
out = write_csv(os.path.join(FINAL, args.name + ".csv"), labels)
digest = sha256(out)
put("submission", {"file": os.path.relpath(out, registry.ROOT),
                   "config": os.path.relpath(os.path.abspath(args.config), registry.ROOT),
                   "config_hash": config.config_hash(cfg), "sha256": digest})

ok = check("official validator on the exact file", validate(out, sp["n_frames"]) is None)
real = not sp["fake"]
pub = cfg["published"]
kicks = {int(k): v for k, v in pub["kickoff_frames"].items()} if real else None
team = {int(k): v for k, v in pub["kickoff_team"].items()} if real else None
b = bands(labels, sp, cfg["bands"], kicks, team, tol=cfg["kickoff"]["tolerance_frames"])
passed = check("structural bands (PLAN.md 6.3)", b["pass"],
               failed=",".join(k for k, v in b["checks"].items() if not v) or "none")
ok &= passed or not real          # the bands describe the real match, not a short synthetic one
for name, v in b.get("advisory", {}).items():
    check("band {} (advisory this cycle, never gates)".format(name), v,
          d_share_graded=b["d_share_graded"])

hash_file = os.path.join(FINAL, args.name + ".sha256")
line = "{}  {}.csv\n".format(digest, args.name)
if real:
    if os.path.isfile(hash_file):
        with open(hash_file) as handle:
            recorded = handle.read().split()[0]
        put("same sha256 as the committed .sha256 file", recorded == digest)
        ok &= recorded == digest
    else:
        with open(hash_file, "w") as handle:
            handle.write(line)
        put("recorded", os.path.relpath(hash_file, registry.ROOT))
sys.exit(0 if ok else 1)
