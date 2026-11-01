"""make airlock T=stability [CFG=...]: self-consistency of a candidate (PLAN.md §6.2c), label-free.

The config's output is the reference. Each numeric threshold in stability.params is moved by
-/+ stability.perturb_frac (integers by at least 1), one at a time, and the pipeline is re-run.
With owner.decoder viterbi, the takeover model is also retrained on the sim's fit matches with
other seeds, and so are the gap models with gaps.enabled. Each variant is compared with the reference on graded frames:
  agree  share of frames with the same label;
  spell  official spellF1 of the variant scored against the reference as if it were the GT.
A candidate whose spells shift wholesale (spell < stability.min_spell) under a 10% change is
fragile. Aggregates only; works on the real match (airlock) and on sim matches.
"""

import copy
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import check, game_args, put  # noqa: E402

from mtah import config, pipeline, registry  # noqa: E402
from mtah.eval.score import pm  # noqa: E402
from mtah.io.atd import load_game  # noqa: E402


def add(parser):
    parser.add_argument("--config", default="configs/final.yaml")


def compare(ref, other, graded):
    agree = float(np.mean(ref[graded] == other[graded]))
    spell = pm.evaluate(pm.label_codes(ref), pm.label_codes(other), bootstrap_n=0)["spell_f1_mean"]
    return round(agree, 4), round(float(spell), 4)


def moved(value, sign, frac):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        step = max(1, int(round(abs(value) * frac)))
        return value + sign * step
    return value * (1 + sign * frac)


def seed_models(cfg, seeds, folder):
    """{seed: model path} retrained on the sim's fit matches (open data) with that seed."""
    import train_takeover as tt
    fit = [tt.Prepared("DFL-" + g, cfg) for g in cfg["matches"]["fit"]]
    out = {}
    for seed in seeds:
        booster = tt.fit_on(fit, dict(cfg["takeover_model"], seed=seed))
        out[seed] = os.path.join(folder, "takeover_seed{}.txt".format(seed))
        booster.save_model(out[seed])
    return out


def seed_gap_models(cfg, seeds, folder):
    """{seed: {"dead": path, "owner": path}}: gap models retrained on the sim's fit matches."""
    import train_gaps as tg
    import train_takeover as tt
    from mtah.learn import gapstate
    gs = [tg.Gaps(tt.Prepared("DFL-" + g, cfg), cfg["gaps"]) for g in cfg["matches"]["fit"]]
    names = gapstate.feature_names(cfg["gaps"])
    out = {}
    for seed in seeds:
        out[seed] = {}
        for k, booster in tg.fit_gaps(gs, dict(cfg["gap_model"], seed=seed), names).items():
            out[seed][k] = os.path.join(folder, "gap_{}_seed{}.txt".format(k, seed))
            booster.save_model(out[seed][k])
    return out


args = game_args(add=add)
cfg = config.load(os.path.abspath(args.config))
st = cfg["stability"]
sp = registry.spec(args.game, args.data_dir)
graded = registry.graded_mask(sp)
atd, _ = load_game(args.game, args.data_dir)
ref, _ = pipeline.run(args.game, cfg, args.data_dir, atd=atd)

rows, spells = {}, []
for section, name in (p.split(".") for p in st["params"]):
    if name not in cfg.get(section, {}):
        continue
    row = []
    for sign in (-1, 1):
        value = moved(cfg[section][name], sign, st["perturb_frac"])
        if value is None:
            continue
        variant = copy.deepcopy(cfg)
        variant[section][name] = value
        labels, _ = pipeline.run(args.game, variant, args.data_dir, atd=atd)
        agree, spell = compare(ref, labels, graded)
        row += [agree, spell]
        spells.append(spell)
    rows["{}.{}".format(section, name)] = row
put("perturbed -/+{:.0%}: [agree-, spell-, agree+, spell+]".format(st["perturb_frac"]), rows)

if cfg["owner"]["decoder"] == "viterbi" and st["seeds"]:
    with tempfile.TemporaryDirectory() as folder:
        seeds = {}
        for seed, path in seed_models(cfg, st["seeds"], folder).items():
            variant = copy.deepcopy(cfg)
            variant["owner"]["model"] = path
            labels, _ = pipeline.run(args.game, variant, args.data_dir, atd=atd)
            agree, spell = compare(ref, labels, graded)
            seeds["seed {}".format(seed)] = [agree, spell]
            spells.append(spell)
    put("takeover model retrained with other seeds: [agree, spell]", seeds)

if cfg.get("gaps", {}).get("enabled") and st["seeds"]:
    with tempfile.TemporaryDirectory() as folder:
        seeds = {}
        for seed, paths in seed_gap_models(cfg, st["seeds"], folder).items():
            variant = copy.deepcopy(cfg)
            for k, path in paths.items():
                variant["gaps"][k + "_model"] = path
            labels, _ = pipeline.run(args.game, variant, args.data_dir, atd=atd)
            agree, spell = compare(ref, labels, graded)
            seeds["seed {}".format(seed)] = [agree, spell]
            spells.append(spell)
    put("gap models retrained with other seeds: [agree, spell]", seeds)

put("spell agreement over all variants (min, median)",
    [float(np.min(spells)), float(np.median(spells))])
check("no wholesale shift under a 10% change (min spell agreement >= {})".format(st["min_spell"]),
      min(spells) >= st["min_spell"], min=float(np.min(spells)))
