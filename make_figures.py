"""Editorial-grade figures for the DA4022 PENCIL compute-allocation study.

- python make_figures.py <tag>   -> results/<tag>/fig.png (2-panel per-setting figure)
- python make_figures.py         -> builds all 4 per-setting figures AND the 3-panel
                                    master figure results/fig_mid.png for the 1-slide report.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Editorial palette (aligned with PENCIL_Devils_Advocate deck)
BG       = "#FAF8F5"
PANEL_BG = "#FFFFFF"
INK      = "#12161C"
SLATE    = "#333B45"
MUTED    = "#6E7781"
RULE     = "#E2E6EA"
CRIMSON  = "#9D1B32"
NAVY     = "#1F3A5F"
TEAL     = "#14746F"
AMBER    = "#C06014"
GOLD     = "#9A7B1C"
GREY     = "#7B848E"
PLUM     = "#6C3B5E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#B8C0C8",
    "axes.linewidth": 0.8,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": INK,
})

LABEL = {
    "d2_f20": "s = 2×20 (official)",
    "d2_f10": "s = 2×10 (half fanout)",
    "d2_f5":  "s = 2×5",
    "d2_f2":  "s = 2×2",
    "d1_f20": "s = 1×20 (1-hop)",
    "d3_f5":  "s = 3×5",
}
SHORT = {
    "d2_f20": "2×20 (full)",
    "d2_f10": "2×10",
    "d2_f5":  "2×5",
    "d2_f2":  "2×2",
    "d1_f20": "1×20 (1-hop)",
    "d3_f5":  "3×5",
}
COLOR = {
    "d2_f20": NAVY,
    "d2_f10": TEAL,
    "d2_f5":  AMBER,
    "d2_f2":  GOLD,
    "d1_f20": GREY,
    "d3_f5":  PLUM,
}


def load_tag(tag):
    R = list(csv.DictReader(open(os.path.join(RES, tag, "ksweep.csv"))))
    S = json.load(open(os.path.join(RES, tag, "summary.json")))
    cost = {k.split("|")[0]: v["fwd_ms_per_cand"]
            for k, v in S["cost"].items() if k.endswith("|test")}
    fracs = sorted({float(r["frac"]) for r in R})
    return R, S, cost, fracs


def series(R, fracs, budget, key="mrr", split="test"):
    m, s = [], []
    for f in fracs:
        v = [float(r[key]) * 100 for r in R
             if r["split"] == split and r["budget"] == budget and abs(float(r["frac"]) - f) < 1e-9]
        m.append(np.mean(v))
        s.append(np.std(v))
    return np.array(m), np.array(s)


def style_ax(ax, bg=PANEL_BG):
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0C8")
    ax.spines["bottom"].set_color("#B8C0C8")
    ax.grid(axis="y", color=RULE, linestyle="-", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.5, length=3, color="#B8C0C8")


def pareto_points(R, cost, fracs):
    full = cost["d2_f20"]
    pts = []
    for b in cost:
        m, s = series(R, fracs, b)
        for f, mm, ss in zip(fracs, m, s):
            pts.append((f * cost[b] / full, mm, ss, b, f))
    pts.sort(key=lambda t: (t[0], -t[1]))
    front, best = [], -1
    for t in pts:
        if t[0] > 0 and t[1] > best + 1e-9:
            front.append(t)
            best = t[1]
    return pts, front


def build_single(tag):
    ds, proto = tag.split("_")
    R, S, cost, fracs = load_tag(tag)
    full = cost["d2_f20"]
    nseeds = len({r["seed"] for r in R})
    pname = {"pool": "shared negative pool", "heart": "HeaRT 500 hard negatives/query"}[proto]
    xlab = {"pool": "Shortlist re-ranked by PENCIL, k / N (%)",
            "heart": "Per-query candidates re-ranked by PENCIL, k / 501 (%)"}[proto]

    full_m = series(R, fracs, "d2_f20")[0][-1]
    top = max(max(series(R, fracs, b)[0].max() for b in cost), full_m) * 1.18
    aa = S["baselines"]["test"]["AA_tiebreak"]["mrr"] * 100
    chance = S["baselines"]["test"]["chance_mrr"] * 100

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.1), dpi=240, facecolor=BG)
    style_ax(a1)
    style_ax(a2)

    x = np.array(fracs) * 100
    for b in [b for b in ["d2_f20", "d2_f10", "d2_f5", "d2_f2", "d1_f20"] if b in cost]:
        m, s = series(R, fracs, b)
        lw = 2.2 if b in ("d2_f20", "d2_f10", "d2_f5") else 1.5
        a1.plot(x, m, "-o", ms=3.2, lw=lw, color=COLOR[b], label=LABEL[b], zorder=4)
        a1.fill_between(x, m - s, m + s, color=COLOR[b], alpha=0.12, lw=0, zorder=2)

    rm, _ = series(R, fracs, "d2_f20", "rand_mrr")
    a1.plot(x, rm, ":", color=INK, lw=1.2, label="Random shortlist (control)", zorder=3)
    a1.axhline(aa, color=CRIMSON, ls="--", lw=1.1, label=f"Adamic-Adar alone ({aa:.1f})", zorder=3)
    a1.axhline(chance, color=MUTED, ls="-.", lw=0.8, label=f"Chance ({chance:.1f})", zorder=2)

    a1.set_xlabel(xlab, fontsize=9, color=SLATE, labelpad=4)
    a1.set_ylabel("Test MRR (x100)", fontsize=9, color=SLATE)
    a1.set_title(f"A. Shortlist Sweep  ({ds.capitalize()} · {pname} · {nseeds} seeds)",
                 fontsize=10, fontweight="bold", color=INK, loc="left", pad=8)
    a1.set_ylim(0, top)
    a1.set_xlim(-2, 103)
    a1.legend(fontsize=7.2, loc="lower right", frameon=True, facecolor="#FFFFFF",
              edgecolor=RULE, framealpha=0.92)

    pts, front = pareto_points(R, cost, fracs)
    for b in cost:
        p = [t for t in pts if t[3] == b and t[0] > 0]
        a2.scatter([t[0] * 100 for t in p], [t[1] for t in p], s=22, color=COLOR[b],
                   alpha=0.82, edgecolors="white", linewidths=0.4, label=LABEL[b], zorder=4)
    a2.plot([t[0] * 100 for t in front], [t[1] for t in front], "-", color=CRIMSON,
            lw=1.8, label="Pareto frontier", zorder=5)
    a2.axhline(full_m, color=NAVY, ls=":", lw=0.9, zorder=2)
    a2.axhline(aa, color=CRIMSON, ls="--", lw=1.0, zorder=2)
    a2.set_xscale("log")
    a2.set_xlabel("Measured GPU compute (% of official PENCIL, log scale)", fontsize=9, color=SLATE, labelpad=4)
    a2.set_ylabel("Test MRR (x100)", fontsize=9, color=SLATE)
    a2.set_title(f"B. Accuracy-Compute Pareto Frontier  ({ds.capitalize()})",
                 fontsize=10, fontweight="bold", color=INK, loc="left", pad=8)
    a2.set_ylim(0, top)
    a2.legend(fontsize=7.0, loc="lower right", frameon=True, facecolor="#FFFFFF",
              edgecolor=RULE, framealpha=0.92)

    fig.tight_layout(pad=1.1)
    out = os.path.join(RES, tag, "fig.png")
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", out)


def build_master_slide_fig():
    """3-panel editorial figure tailored to the 1-slide mid-project report."""
    Rc, Sc, cost_c, fr_c = load_tag("cora_pool")
    Rs, Ss, cost_s, fr_s = load_tag("citeseer_pool")
    Rch, Sch, cost_ch, fr_ch = load_tag("cora_heart")
    Rsh, Ssh, cost_sh, fr_sh = load_tag("citeseer_heart")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.85), dpi=300, facecolor=BG)
    for ax in (ax1, ax2, ax3):
        style_ax(ax)

    # --- PANEL A: Cora shortlist sweep (k/N) showing why top-10% is flat & 1-hop collapses ---
    x = np.array(fr_c) * 100
    ax1.axvspan(0, 10, color="#F3E8EA", alpha=0.75, zorder=1)
    ax1.text(5.2, 46.8, "AA already\nsaturated\n(k ≤ 10%)", ha="center", va="top",
             fontsize=6.8, color=CRIMSON, fontweight="bold")

    for b, lw in [("d2_f20", 2.3), ("d2_f10", 2.1), ("d2_f5", 2.0), ("d2_f2", 1.5), ("d1_f20", 1.6)]:
        m, s = series(Rc, fr_c, b)
        ls = "--" if b == "d1_f20" else "-"
        ax1.plot(x, m, ls, marker="o", ms=3.0, lw=lw, color=COLOR[b], label=SHORT[b], zorder=4)
        ax1.fill_between(x, m - s, m + s, color=COLOR[b], alpha=0.11, lw=0, zorder=2)

    rm, _ = series(Rc, fr_c, "d2_f20", "rand_mrr")
    ax1.plot(x, rm, ":", color=INK, lw=1.2, label="Random shortlist", zorder=3)
    aa_c = Sc["baselines"]["test"]["AA_tiebreak"]["mrr"] * 100
    ax1.axhline(aa_c, color=CRIMSON, ls="--", lw=1.0, zorder=3)
    ax1.text(99, aa_c - 1.3, f"AA alone ({aa_c:.1f})", ha="right", va="top",
             fontsize=7.2, color=CRIMSON, fontweight="bold")

    ax1.annotate("1-hop peaks at k=20%\nthen hurts (35.2 → 20.3)",
                 xy=(30, 18.0), xytext=(36, 9.0),
                 fontsize=7.0, color=SLATE,
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8, connectionstyle="arc3,rad=-0.15"))

    ax1.set_title("A. Shortlist Sweep on Cora (k / N)", fontsize=9.8, fontweight="bold",
                  color=INK, loc="left", pad=7)
    ax1.set_xlabel("Candidates re-ranked by PENCIL, k / N (%)", fontsize=8.3, color=SLATE)
    ax1.set_ylabel("Test MRR (×100)", fontsize=8.5, color=SLATE)
    ax1.set_xlim(-2, 103)
    ax1.set_ylim(0, 49)
    ax1.legend(fontsize=6.8, loc="lower right", frameon=True, facecolor="#FFFFFF",
               edgecolor=RULE, ncol=2, columnspacing=0.8, handlelength=1.5)

    # --- PANEL B: Pareto Frontier on BOTH Cora & Citeseer (Pool protocol) ---
    pts_c, front_c = pareto_points(Rc, cost_c, fr_c)
    pts_s, front_s = pareto_points(Rs, cost_s, fr_s)

    for t in pts_c:
        if t[0] > 0:
            ax2.scatter(t[0] * 100, t[1], s=18, color=NAVY, alpha=0.28, edgecolors="none", zorder=2)
    for t in pts_s:
        if t[0] > 0:
            ax2.scatter(t[0] * 100, t[1], s=18, color=TEAL, alpha=0.28, marker="s", edgecolors="none", zorder=2)

    ax2.plot([t[0] * 100 for t in front_c], [t[1] for t in front_c], "-o", ms=3.8,
             color=NAVY, lw=2.2, label="Cora Pareto frontier", zorder=5)
    ax2.plot([t[0] * 100 for t in front_s], [t[1] for t in front_s], "-s", ms=3.8,
             color=TEAL, lw=2.2, label="Citeseer Pareto frontier", zorder=5)

    # Highlight validation-picked sweet spots
    # Cora 7% compute -> 38.4; Cora 67% -> 42.9; Citeseer 63% -> 51.4 (beats 100% -> 49.7)
    ax2.scatter([7.2], [38.4], s=55, color=CRIMSON, zorder=6, edgecolors="white", linewidths=0.8)
    ax2.annotate("Cora @ 7% GPU\n38.4 MRR (+5.3 vs AA)",
                 xy=(7.2, 38.4), xytext=(1.3, 41.5),
                 fontsize=7.0, color=CRIMSON, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.9))

    ax2.scatter([62.8], [51.4], s=55, color=TEAL, zorder=6, edgecolors="white", linewidths=0.8)
    ax2.scatter([100.0], [49.7], s=42, color=INK, zorder=6, marker="x", linewidths=1.3)
    ax2.annotate("Citeseer s=2×10 (63% GPU)\n51.4 MRR > full 49.7!",
                 xy=(62.8, 51.4), xytext=(3.5, 51.0),
                 fontsize=7.0, color=TEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))

    ax2.axhline(aa_c, color=NAVY, ls=":", lw=0.8, alpha=0.7)
    ax2.axhline(Ss["baselines"]["test"]["AA_tiebreak"]["mrr"] * 100, color=TEAL, ls=":", lw=0.8, alpha=0.7)
    ax2.text(0.18, 33.6, "AA baselines (33.1 / 31.5)", fontsize=6.8, color=MUTED)

    ax2.set_xscale("log")
    ax2.set_title("B. Compute Pareto Frontier (Log GPU Time)", fontsize=9.8, fontweight="bold",
                  color=INK, loc="left", pad=7)
    ax2.set_xlabel("Measured T4 GPU time (% of official s=2×20, log)", fontsize=8.3, color=SLATE)
    ax2.set_ylabel("Test MRR (×100)", fontsize=8.5, color=SLATE)
    ax2.set_ylim(26, 56)
    ax2.set_xlim(0.12, 135)
    ax2.legend(fontsize=7.2, loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=RULE)

    # --- PANEL C: HeaRT Hard-Negative Regime (500 hard negatives per query) ---
    # Show grouped horizontal or clean slope/bar comparison of the 4 regimes across Both Graphs & Both Protocols,
    # OR show the HeaRT Pareto curves for Cora & Citeseer!
    _, front_ch = pareto_points(Rch, cost_ch, fr_ch)
    _, front_sh = pareto_points(Rsh, cost_sh, fr_sh)

    # Plot full s=2x20 and s=2x10 and s=2x5 curves on Citeseer HeaRT & Cora HeaRT vs compute
    for tag_lbl, R_h, cost_h, fr_h, col, ls_main in [
        ("Citeseer HeaRT", Rsh, cost_sh, fr_sh, CRIMSON, "-"),
        ("Cora HeaRT", Rch, cost_ch, fr_ch, NAVY, "--"),
    ]:
        full_h = cost_h["d2_f20"]
        m20, _ = series(R_h, fr_h, "d2_f20")
        c20 = [f * 100 for f in fr_h if f > 0]
        ax3.plot(c20, m20[1:], ls_main, marker="o", ms=3.0, lw=2.0, color=col,
                 label=f"{tag_lbl} (s=2×20)")
        m10, _ = series(R_h, fr_h, "d2_f10")
        c10 = [f * cost_h["d2_f10"] / full_h * 100 for f in fr_h if f > 0]
        ax3.plot(c10, m10[1:], ":", marker="^", ms=2.8, lw=1.5, color=col, alpha=0.85,
                 label=f"{tag_lbl} (s=2×10)")

    aa_ch = Sch["baselines"]["test"]["AA_tiebreak"]["mrr"] * 100
    aa_sh = Ssh["baselines"]["test"]["AA_tiebreak"]["mrr"] * 100
    ax3.axhline(aa_ch, color=NAVY, ls=":", lw=0.8, alpha=0.6)
    ax3.axhline(aa_sh, color=CRIMSON, ls=":", lw=0.8, alpha=0.6)
    ax3.text(0.22, 11.2, "AA on HeaRT (12.0 / 11.0)", fontsize=6.8, color=MUTED)

    ax3.annotate("Citeseer HeaRT needs full s=2×20\n(17.8 vs 14.2 at s=2×10)",
                 xy=(100, 17.76), xytext=(2.0, 17.5),
                 fontsize=6.9, color=CRIMSON, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.9))
    ax3.annotate("Cora HeaRT: top-1% @ s=2×5\ngets 12.8 MRR at 0.4% GPU",
                 xy=(0.38, 12.8), xytext=(0.5, 15.1),
                 fontsize=6.8, color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))

    ax3.set_xscale("log")
    ax3.set_title("C. Hard Negatives (HeaRT, 500/query)", fontsize=9.8, fontweight="bold",
                  color=INK, loc="left", pad=7)
    ax3.set_xlabel("Measured T4 GPU time (% of official s=2×20, log)", fontsize=8.3, color=SLATE)
    ax3.set_ylabel("Test MRR (×100)", fontsize=8.5, color=SLATE)
    ax3.set_ylim(9.5, 19.5)
    ax3.set_xlim(0.15, 135)
    ax3.legend(fontsize=6.6, loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=RULE)

    fig.tight_layout(pad=1.05)
    out = os.path.join(RES, "fig_mid.png")
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote master slide figure:", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        build_single(sys.argv[1])
        if sys.argv[1] == "cora_pool":
            build_master_slide_fig()
    else:
        for t in ("cora_pool", "citeseer_pool", "cora_heart", "citeseer_heart"):
            if os.path.exists(os.path.join(RES, t, "ksweep.csv")):
                build_single(t)
        build_master_slide_fig()
