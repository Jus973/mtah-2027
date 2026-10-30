import sys, numpy as np, pandas as pd
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
import possession_metrics as pm

gt = pm.label_codes(pm.read_label_csv(os.path.join(HERE, "..", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
segs = pm.segments(gt, runs)

def score(pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]

def shifted(k):
    p = gt.copy()
    for a,b in runs:
        w = gt[a:b]
        s = np.concatenate((np.full(k, w[0]), w[:-k])) if k>0 else w
        p[a:b] = s
    return p

print("perfect:", score(gt.copy()))
for k in (3, 8, 12, 25, 50):
    print(f"all boundaries {k} frames late ({k/25:.2f}s):", score(shifted(k)))

# merge spells shorter than T seconds into previous label
def drop_short(T):
    p = gt.copy()
    for (a,b,c) in segs:
        if c in (pm.H, pm.A) and (b-a) < T*25:
            # assign to previous label (the label right before a)
            prev = p[a-1] if a>0 and gt[a-1]!=pm.X else None
            if prev is not None:
                p[a:b] = prev
    return p
for T in (1.0, 2.0, 4.0):
    print(f"blips < {T}s merged into previous spell:", score(drop_short(T)))

# miss all D: replace D with previous label
def no_dead():
    p = gt.copy()
    for (a,b,c) in segs:
        if c == pm.D and a>0 and gt[a-1] != pm.X:
            p[a:b] = p[a-1]
    return p
print("every D predicted as continuing possession:", score(no_dead()))

# spurious D of 1s in the middle of each spell longer than 10 s
def spurious_D(sec=1.0):
    p = gt.copy()
    for (a,b,c) in segs:
        if c in (pm.H, pm.A) and (b-a) > 250:
            m = (a+b)//2; p[m:m+int(sec*25)] = pm.D
    return p
print("spurious 1s D inside each long spell:", score(spurious_D()))

# one extra turnover flicker of 0.5 s inside each long spell
def flicker(sec=0.5):
    p = gt.copy()
    for (a,b,c) in segs:
        if c in (pm.H, pm.A) and (b-a) > 250:
            m = (a+b)//2; p[m:m+int(sec*25)] = pm.A if c==pm.H else pm.H
    return p
print("spurious 0.5s opponent flicker inside each long spell:", score(flicker()))

# late flips only at turnovers of 1s, D boundaries exact
def late_turnovers(k):
    p = gt.copy()
    for i in range(1, len(segs)):
        a0,b0,c0 = segs[i-1]; a1,b1,c1 = segs[i]
        if c0 in (pm.H,pm.A) and c1 in (pm.H,pm.A):
            p[a1:a1+k] = c0
    return p
for k in (12, 25, 50):
    print(f"turnover flips {k} frames late, D exact:", score(late_turnovers(k)))
