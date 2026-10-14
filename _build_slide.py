"""Builds the 1-slide mid-project report: MIDREPORT.pptx (+ PDF/PNG via PowerPoint)."""
import csv, json, os
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
R = list(csv.DictReader(open(os.path.join(RES, "cora_pool", "ksweep.csv"))))
S = json.load(open(os.path.join(RES, "cora_pool", "summary.json")))
COST = {k.split("|")[0]: v["fwd_ms_per_cand"] for k, v in S["cost"].items() if k.endswith("|test")}


def cell(budget, frac, split="test"):
    v = [float(r["mrr"]) * 100 for r in R if r["split"] == split and r["budget"] == budget
         and abs(float(r["frac"]) - frac) < 1e-9]
    return np.mean(v), np.std(v), frac * COST[budget] / COST["d2_f20"] * 100


NAVY, GREY, RED = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0x40, 0x40, 0x40), RGBColor(0xA5, 0x00, 0x21)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])


def box(x, y, w, h):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf


def para(tf, text, size=11, bold=False, color=GREY, first=False, space=2, runs=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space)
    for t, b, c in (runs or [(text, bold, color)]):
        r = p.add_run(); r.text = t
        r.font.size, r.font.bold, r.font.color.rgb = Pt(size), b, c
    return p


tf = box(0.35, 0.18, 12.6, 0.6)
para(tf, "How Much PENCIL Can You Afford?  Compute-allocation for link prediction", 24, True, NAVY,
     first=True)
tf = box(0.35, 0.78, 12.6, 0.35)
from team import team_line
para(tf, "DA4022 mid-project update  |  " + team_line(),
     11, color=GREY, first=True)

tf = box(0.35, 1.25, 4.75, 6.1)
para(tf, "WHAT WE BUILT", 12.5, True, NAVY, first=True)
for t in ["Reproduced PENCIL on one free T4: test MRR 44.0 (paper: 42.2).",
          "Retrieve-then-rerank: Adamic–Adar (AA) ranks every candidate; PENCIL re-scores only the top k.",
          "Two inference knobs, no retraining: k = how many re-ranked; s = sampling budget per "
          "candidate (hops × neighbours).",
          "Cora: 6 settings of s × 3 seeds × 12 values of k. Cost = measured T4 forward time.",
          "Checks: our AA/CN reproduce the HeaRT benchmark exactly (31.87 / 20.99); our metric "
          "matches the repo evaluator (Δ < 10⁻⁷)."]:
    para(tf, "• " + t, 11)
para(tf, "", 4)
para(tf, "FINDINGS SO FAR", 12.5, True, NAVY)
for head, rest in [("Cheap regime: ", "re-ranking AA's top 20% at s = 2×5 costs 7% of full PENCIL and "
                    "adds +5 MRR over AA on both splits."),
                   ("Halving s is free: ", "s = 2×10 matches full PENCIL on test and validation at 67% "
                    "of the compute. s = 2×5 is cheaper but unstable (seed std up to 4.1)."),
                   ("Re-ranking the top 10% does nothing: ",
                    "AA's top picks are already true links. PENCIL earns its cost only where AA is "
                    "blind: 57% of true test links share no neighbour."),
                   ("k and s interact: ", "a weak re-ranker (1 hop) peaks at k = 20% and then hurts "
                    "(35 → 20 MRR); the full one keeps improving.")]:
    para(tf, "", 11, runs=[("• " + head, True, GREY), (rest, False, GREY)])
para(tf, "", 4)
para(tf, "NEXT (→ Nov 26)", 12.5, True, NAVY)
for t in ["citeseer + ogbl-collab, and HeaRT per-query hard negatives.",
          "Fit a (k, s) rule on validation; test it on a graph it never saw.",
          "Proper latency and memory benchmark (repeats, batch size 1).",
          "Training axis: 93% of final MRR is reached by epoch 42 of 150."]:
    para(tf, "• " + t, 11)

s.shapes.add_picture(os.path.join(RES, "fig_mid.png"), Inches(5.25), Inches(1.3), width=Inches(7.85))

rows = [("Setting (cora)", "test MRR", "valid MRR", "compute")]
m0, _, _ = cell("d2_f20", 0.0)
mv0, _, _ = cell("d2_f20", 0.0, "valid")
rows.append(("Adamic–Adar alone (degree tie-break)", f"{m0:.1f}", f"{mv0:.1f}", "0%"))
for label, b, f in [("AA top-20% → PENCIL, s = 2×5", "d2_f5", 0.20),
                    ("All → PENCIL, s = 2×10", "d2_f10", 1.0),
                    ("All → PENCIL, s = 2×5", "d2_f5", 1.0),
                    ("Full PENCIL (official s = 2×20)", "d2_f20", 1.0)]:
    m, sd, c = cell(b, f)
    mv, sdv, _ = cell(b, f, "valid")
    rows.append((label, f"{m:.1f} ± {sd:.1f}", f"{mv:.1f} ± {sdv:.1f}", f"{c:.0f}%"))
HL = {2, 3}
tbl = s.shapes.add_table(len(rows), 4, Inches(5.45), Inches(4.2), Inches(7.6), Inches(1.9)).table
for j, wdt in enumerate((3.3, 1.5, 1.5, 1.3)):
    tbl.columns[j].width = Inches(wdt)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = val
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        f = p.runs[0].font; f.size = Pt(10.5); f.bold = (i == 0 or i in HL)
        if i in HL: f.color.rgb = RED

tf = box(5.45, 6.2, 7.6, 1.2)
para(tf, "Protocol: cora planetoid split; each of 527 test links ranked against all 527 test "
     "negatives (chance MRR = 1.3); validation 263 vs 263. Compute = measured T4 forward time vs "
     "full PENCIL. Settings above are read off a sweep, not yet selected on validation alone — the "
     "final report will. One graph so far; timings from a single pass.",
     9.5, color=GREY, first=True)

out = os.path.join(HERE, "docs", "MIDREPORT_public.pptx") if os.environ.get("PENCIL_PUBLIC") \
    else os.path.join(HERE, "MIDREPORT.pptx")
prs.save(out)
print("wrote", out)
