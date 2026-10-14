"""Two-panel budget figure for one dataset/protocol.

python make_figures.py [cora_pool|cora_heart|citeseer_pool|citeseer_heart]
-> results/<tag>/fig.png   (cora_pool is also copied to results/fig_mid.png for the slide)
"""
import csv, json, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "cora_pool"
DS, PROTO = TAG.split("_")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
R = list(csv.DictReader(open(os.path.join(RES, TAG, "ksweep.csv"))))
S = json.load(open(os.path.join(RES, TAG, "summary.json")))
SPLIT = "test"
COST = {k.split("|")[0]: v["fwd_ms_per_cand"] for k, v in S["cost"].items() if k.endswith("|" + SPLIT)}
FULL = COST["d2_f20"]
FRACS = sorted({float(r["frac"]) for r in R})
NSEEDS = len({r["seed"] for r in R})
LABEL = {"d2_f20": "s = 2 hops x 20 (official)", "d2_f10": "s = 2 hops x 10",
         "d2_f5": "s = 2 hops x 5", "d2_f2": "s = 2 hops x 2",
         "d1_f20": "s = 1 hop x 20", "d3_f5": "s = 3 hops x 5"}
COLOR = {"d2_f20": "#1f4e79", "d2_f10": "#2e75b6", "d2_f5": "#c55a11", "d2_f2": "#bf9000",
         "d1_f20": "#7f7f7f", "d3_f5": "#548235"}
XLAB = {"pool": "% of candidate pool re-ranked by PENCIL  (k / N)",
        "heart": "% of each query's 501 candidates re-ranked by PENCIL  (k / 501)"}[PROTO]
PNAME = {"pool": "shared pool", "heart": "HeaRT hard negatives"}[PROTO]


def series(budget, key="mrr"):
    m, s = [], []
    for f in FRACS:
        v = [float(r[key]) * 100 for r in R if r["split"] == SPLIT and r["budget"] == budget
             and float(r["frac"]) == f]
        m.append(np.mean(v)); s.append(np.std(v))
    return np.array(m), np.array(s)


full_m = series("d2_f20")[0][-1]
top = max(max(series(b)[0].max() for b in COST), full_m) * 1.18
aa = S["baselines"][SPLIT]["AA"]["mrr"] * 100
chance = S["baselines"][SPLIT]["chance_mrr"] * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), dpi=200)
x = np.array(FRACS) * 100
for b in [b for b in ["d2_f20", "d2_f10", "d2_f5", "d2_f2", "d1_f20"] if b in COST]:
    m, s = series(b)
    a1.plot(x, m, "-o", ms=3, lw=1.8, color=COLOR[b], label=LABEL[b])
    a1.fill_between(x, m - s, m + s, color=COLOR[b], alpha=0.15, lw=0)
rm, _ = series("d2_f20", "rand_mrr")
a1.plot(x, rm, ":", color="black", lw=1.2, label="control: random shortlist, s official")
a1.axhline(aa, color="#a50021", ls="--", lw=1, label=f"Adamic-Adar alone, ties averaged ({aa:.1f})")
a1.axhline(chance, color="gray", ls="-.", lw=0.8, label=f"chance ({chance:.1f})")
a1.set_xlabel(XLAB); a1.set_ylabel("test MRR (x100)")
a1.set_title(f"Re-rank how many?  ({DS}, {PNAME}, mean +/- std, {NSEEDS} seeds)", fontsize=10)
a1.set_ylim(0, top); a1.grid(alpha=0.3); a1.legend(fontsize=6.6, loc="lower right")

pts = []
for b in COST:
    m, s = series(b)
    pts += [(f * COST[b] / FULL, mm, ss, b, f) for f, mm, ss in zip(FRACS, m, s)]
for b in COST:
    p = [t for t in pts if t[3] == b and t[0] > 0]
    a2.scatter([t[0] * 100 for t in p], [t[1] for t in p], s=14, color=COLOR[b], alpha=0.75,
               label=LABEL[b])
pts.sort(key=lambda t: (t[0], -t[1]))
front, best = [], -1
for t in pts:
    if t[1] > best + 1e-9 and t[0] > 0:
        front.append(t); best = t[1]
a2.plot([t[0] * 100 for t in front], [t[1] for t in front], "k-", lw=1.2, label="Pareto frontier")
a2.axhline(full_m, color="#1f4e79", ls=":", lw=0.9)
a2.annotate(f"full PENCIL\n{full_m:.1f} @ 100%", (100, full_m), (30, full_m + top * 0.06),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
cheap = [t for t in front if t[0] <= 0.10]
if cheap:
    pick = max(cheap, key=lambda t: t[1])
    a2.annotate(f"{pick[1]:.1f} MRR @ {pick[0]*100:.0f}% compute\n(top {pick[4]*100:.0f}% by AA, "
                f"{LABEL[pick[3]].split(' (')[0]})", (pick[0] * 100, pick[1]),
                (pick[0] * 100 * 1.4, top * 0.22), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
a2.axhline(aa, color="#a50021", ls="--", lw=1)
a2.set_xscale("log")
a2.set_xlabel("PENCIL compute, % of full model (measured T4 forward time, log)")
a2.set_ylabel("test MRR (x100)")
a2.set_title("Accuracy vs compute: every (k, s) we tried", fontsize=10)
a2.set_ylim(0, top); a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=6.3, loc="lower right")

fig.tight_layout()
out = os.path.join(RES, TAG, "fig.png")
fig.savefig(out, bbox_inches="tight")
if TAG == "cora_pool":
    shutil.copy(out, os.path.join(RES, "fig_mid.png"))
print("wrote", out)
for t in front:
    print(f"  frontier: cost {t[0]*100:5.1f}%  MRR {t[1]:.1f} +/- {t[2]:.1f}  {t[3]} k/N={t[4]:.3f}")
