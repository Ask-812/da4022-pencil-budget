"""Builds the 1-slide mid-project report: MIDREPORT.pptx (+ PDF via PowerPoint).

Designed in the editorial visual language of the team's PENCIL Devil's Advocate deck:
- Warm editorial paper canvas (#FAF8F5) with a full-height Deep Crimson left spine (#9D1B32)
- Georgia Bold display serif headline + Segoe UI precision body + Consolas tabular numerals
- Custom vector schematic for the 2-stage retrieve-then-rerank pipeline
- Numbered editorial findings (01-04) with hairline separators (zero generic bullet dumps)
- 3-panel despined Tufte/Economist matplotlib figure (results/fig_mid.png)
- Custom Booktabs benchmark table across all 4 regimes (Cora/Citeseer x Pool/HeaRT)
"""
import json, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from team import NAMES, ROLLS

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")
SEL  = json.load(open(os.path.join(RES, "selection.json")))

# Editorial color palette
C_BG       = RGBColor(0xFA, 0xF8, 0xF5)  # warm paper
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_IVORY    = RGBColor(0xF5, 0xEF, 0xE4)  # warm callout tint
C_SLATE_BG = RGBColor(0xF0, 0xF3, 0xF6)  # cool card tint
C_ROSE_BG  = RGBColor(0xF9, 0xF1, 0xF2)  # soft crimson tint
C_INK      = RGBColor(0x12, 0x16, 0x1C)  # primary ink
C_SLATE    = RGBColor(0x33, 0x3B, 0x45)  # body text
C_MUTED    = RGBColor(0x6E, 0x77, 0x81)  # captions / metadata
C_RULE     = RGBColor(0xDC, 0xE1, 0xE7)  # hairline rule
C_CRIMSON  = RGBColor(0x9D, 0x1B, 0x32)  # signature accent
C_NAVY     = RGBColor(0x1F, 0x3A, 0x5F)  # official PENCIL accent
C_TEAL     = RGBColor(0x14, 0x74, 0x6F)  # sweet-spot s=2x10 accent
C_AMBER    = RGBColor(0xB8, 0x5D, 0x14)  # frugal hybrid accent

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.500)
s = prs.slides.add_slide(prs.slide_layouts[6])


def rect(x, y, w, h, fill=None, line=None, line_pt=0.75, rounded=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        try:
            sh.adjustments[0] = 0.08
        except Exception:
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_pt)
    return sh


def tbox(x, y, w, h, margin=0.0):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    return tf


def add_p(tf, runs, align=PP_ALIGN.LEFT, space_after=0, space_before=0, line_spacing=1.08, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = line_spacing
    for item in runs:
        text, size, bold, color = item[0], item[1], item[2], item[3]
        font_name = item[4] if len(item) > 4 else "Segoe UI"
        italic = item[5] if len(item) > 5 else False
        r = p.add_run()
        r.text = text
        r.font.name = font_name
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return p


# ==============================================================================
# 0. CANVAS & SIGNATURE LEFT SPINE
# ==============================================================================
rect(0, 0, 13.333, 7.500, fill=C_BG)
rect(0, 0, 0.115, 7.500, fill=C_CRIMSON)

# ==============================================================================
# 1. EDITORIAL HEADER & TEAM ROSTER (y = 0.15 .. 0.88)
# ==============================================================================
tf = tbox(0.34, 0.14, 7.60, 0.20)
add_p(tf, [("DA4022 MID-PROJECT REPORT   ·   SELF-PROPOSED (ANCHOR #2)   ·   EXTENDING PENCIL (ICML 2026)",
            8.2, True, C_CRIMSON)], first=True)

tf = tbox(0.34, 0.34, 7.65, 0.40)
add_p(tf, [
    ("How Much PENCIL Can You Afford?", 19.2, True, C_INK, "Georgia"),
    ("   Compute-Allocation Laws for Link Prediction", 11.4, True, C_SLATE, "Segoe UI"),
], first=True)

# Team roster grid on top-right (matching Slide 1 of Devil's Advocate deck)
team_order = [NAMES[0], NAMES[1], NAMES[2], NAMES[3], NAMES[4]]
tx0 = 8.05
tw  = 1.01
for idx, name in enumerate(team_order):
    cx = tx0 + idx * tw
    if idx > 0:
        rect(cx - 0.04, 0.22, 0.01, 0.48, fill=C_RULE)
    parts = name.split(" ")
    tf_m = tbox(cx, 0.19, tw - 0.05, 0.54)
    add_p(tf_m, [(parts[0], 8.8, True, C_INK)], first=True, line_spacing=1.0)
    add_p(tf_m, [(parts[1], 8.5, False, C_SLATE)], space_after=1, line_spacing=1.0)
    roll_txt = ROLLS.get(name, "DA4022")
    add_p(tf_m, [(roll_txt, 7.5, False, C_MUTED, "Consolas")], line_spacing=1.0)

# Header accent rules
rect(0.34, 0.80, 0.78, 0.038, fill=C_CRIMSON)
rect(1.16, 0.818, 11.92, 0.012, fill=C_RULE)

# ==============================================================================
# 2. LEFT COLUMN: ARCHITECTURE, REPRODUCTION & 4 EMPIRICAL LAWS (x = 0.34 .. 4.62)
# ==============================================================================
LX, LW = 0.34, 4.28

# --- Section I: Architecture schematic ---
tf = tbox(LX, 0.90, LW, 0.20)
add_p(tf, [("I.  TWO-STAGE RETRIEVE-THEN-RERANK PIPELINE", 8.3, True, C_CRIMSON)], first=True)

rect(LX, 1.11, LW, 1.12, fill=C_WHITE, line=C_RULE, line_pt=0.8)
rect(LX, 1.11, 0.045, 1.12, fill=C_NAVY)

# 3 pipeline stage boxes inside card
# Box 1: Candidate Pool
rect(LX + 0.12, 1.20, 1.06, 0.54, fill=C_SLATE_BG, line=C_RULE, line_pt=0.75, rounded=True)
tf = tbox(LX + 0.14, 1.23, 1.02, 0.48)
add_p(tf, [("Candidate Pool\n", 7.8, True, C_INK),
           ("N pairs (Pool) or\n501 / query (HeaRT)", 6.6, False, C_MUTED)],
      align=PP_ALIGN.CENTER, first=True, line_spacing=1.02)

# Arrow 1
tf = tbox(LX + 1.19, 1.33, 0.21, 0.25)
add_p(tf, [("→", 11.5, True, C_MUTED)], align=PP_ALIGN.CENTER, first=True)

# Box 2: Stage 1 Adamic-Adar
rect(LX + 1.41, 1.20, 1.24, 0.54, fill=C_IVORY, line=C_AMBER, line_pt=0.9, rounded=True)
tf = tbox(LX + 1.43, 1.23, 1.20, 0.48)
add_p(tf, [("Stage 1 · Adamic–Adar\n", 7.6, True, C_AMBER),
           ("CPU <1 ms · 0% GPU\nranks & shortlists top k", 6.6, False, C_SLATE)],
      align=PP_ALIGN.CENTER, first=True, line_spacing=1.02)

# Arrow 2
tf = tbox(LX + 2.66, 1.33, 0.21, 0.25)
add_p(tf, [("→", 11.5, True, C_CRIMSON)], align=PP_ALIGN.CENTER, first=True)

# Box 3: Stage 2 PENCIL
rect(LX + 2.88, 1.20, 1.30, 0.54, fill=C_SLATE_BG, line=C_NAVY, line_pt=1.0, rounded=True)
tf = tbox(LX + 2.90, 1.23, 1.26, 0.48)
add_p(tf, [("Stage 2 · PENCIL\n", 7.8, True, C_NAVY),
           ("Transformer re-scores\ntop k with budget s", 6.6, False, C_SLATE)],
      align=PP_ALIGN.CENTER, first=True, line_spacing=1.02)

# Two inference knob strips at bottom of architecture card
rect(LX + 0.12, 1.81, 4.06, 0.34, fill=C_BG, line=C_RULE, line_pt=0.5)
tf = tbox(LX + 0.18, 1.84, 3.96, 0.28)
add_p(tf, [
    ("TWO INFERENCE KNOBS (NO RETRAINING):   ", 7.2, True, C_CRIMSON),
    ("k", 7.8, True, C_INK, "Consolas"),
    (" = shortlist re-ranked (0–100%)   ·   ", 7.2, False, C_SLATE),
    ("s", 7.8, True, C_INK, "Consolas"),
    (" = subgraph hops×fanout", 7.2, False, C_SLATE),
], first=True)

# --- Section II: Reproduction & Audit KPI Strip ---
tf = tbox(LX, 2.31, LW, 0.20)
add_p(tf, [("II.  FREE-GPU REPRODUCTION & AUDIT (PAPER USED 4–8× A100)", 8.3, True, C_CRIMSON)], first=True)

kpis = [
    ("44.00", "Cora Test MRR", "Paper: 42.23 · 1×T4, 3.2h", C_NAVY),
    ("50.46", "Citeseer Test MRR", "Paper: 47.51 · 2×T4, 1.0h", C_TEAL),
    ("112 Runs", "4 Regimes Audited", "HeaRT AA exact · Δ<10⁻⁷", C_CRIMSON),
]
kw = 1.38
for idx, (big, mid, sub, col) in enumerate(kpis):
    kx = LX + idx * (kw + 0.07)
    rect(kx, 2.52, kw, 0.74, fill=C_WHITE, line=C_RULE, line_pt=0.8)
    rect(kx, 2.52, kw, 0.04, fill=col)
    tf = tbox(kx + 0.08, 2.58, kw - 0.16, 0.64)
    add_p(tf, [(big, 13.5, True, col, "Georgia")], first=True, line_spacing=1.0)
    add_p(tf, [(mid, 7.8, True, C_INK)], space_before=1, line_spacing=1.0)
    add_p(tf, [(sub, 6.7, False, C_MUTED)], space_before=1, line_spacing=1.0)

# --- Section III: 4 Numbered Findings (Editorial Slide-2 Style) ---
tf = tbox(LX, 3.35, LW, 0.20)
add_p(tf, [("III.  FOUR EMPIRICAL COMPUTE-ALLOCATION FINDINGS", 8.3, True, C_CRIMSON)], first=True)

findings = [
    ("01", "Official s = 2×20 over-reads in 3 of 4 regimes",
     "Halving fanout to s = 2×10 (~65% GPU, 40% less VRAM) matches Cora (42.9 vs 43.7) and beats full PENCIL on Citeseer Pool (51.4 vs 49.7) and Cora HeaRT (14.3 vs 13.3). Only Citeseer HeaRT needs full 2×20 (17.8 vs 14.2)."),
    ("02", "Frugal hybrid (≤10% GPU) beats heuristics in all 4 tests",
     "Choosing (k, s) on validation alone under a 10% compute cap lifts Test MRR over Adamic–Adar everywhere: 33.1 → 38.4 on Cora (7% GPU) and 11.0 → 13.8 on Citeseer HeaRT (8% GPU)."),
    ("03", "Top-10% re-ranking is wasted; PENCIL pays off in the 0-CN tail",
     "AA's top decile is already true links, so re-ranking k ≤ 10% changes nothing. PENCIL earns its FLOPs where AA is blind: 57.3% of true Cora test links share zero common neighbours."),
    ("04", "Cross-graph budget rules beat noisy local validation",
     "Applying Cora's validation-chosen (k, s) unchanged to Citeseer Pool beats Citeseer's own validation pick at every cap ≤ 50% (44.9 vs 37.9 MRR at 25% cap) because small-graph validation is noisy."),
]

fy = 3.57
fh = 0.89
for idx, (num, title, body) in enumerate(findings):
    y0 = fy + idx * fh
    rect(LX, y0, LW, fh - 0.05, fill=C_WHITE, line=C_RULE, line_pt=0.7)
    rect(LX, y0, 0.42, fh - 0.05, fill=C_ROSE_BG)
    rect(LX, y0, 0.04, fh - 0.05, fill=C_CRIMSON)
    tf_n = tbox(LX + 0.04, y0 + 0.22, 0.38, 0.40)
    add_p(tf_n, [(num, 12.5, True, C_CRIMSON, "Georgia")], align=PP_ALIGN.CENTER, first=True)

    tf_b = tbox(LX + 0.50, y0 + 0.05, LW - 0.58, fh - 0.12)
    add_p(tf_b, [(title, 8.8, True, C_INK)], first=True, space_after=1.5, line_spacing=1.02)
    add_p(tf_b, [(body, 7.35, False, C_SLATE)], line_spacing=1.06)

# ==============================================================================
# 3. RIGHT COLUMN: 3-PANEL PARETO FIGURE + BOOKTABS TABLE + ROADMAP (x = 4.78 .. 13.08)
# ==============================================================================
RX, RW = 4.78, 8.30

# --- Section IV: 3-Panel Figure ---
tf = tbox(RX, 0.90, RW, 0.20)
add_p(tf, [("IV.  SHORTLIST SWEEP, PARETO FRONTIER & HARD-NEGATIVE (HeaRT) REGIMES", 8.3, True, C_CRIMSON)], first=True)

rect(RX, 1.11, RW, 2.66, fill=C_WHITE, line=C_RULE, line_pt=0.8)
s.shapes.add_picture(os.path.join(RES, "fig_mid.png"), Inches(RX + 0.04), Inches(1.13), width=Inches(RW - 0.08))

# --- Section V: Custom Booktabs Benchmark & Transfer Table ---
tf = tbox(RX, 3.84, RW, 0.20)
add_p(tf, [("V.  VALIDATION-SELECTED OPERATING POINTS & CROSS-GRAPH TRANSFER  (TEST MRR ×100, MEAN ± STD)",
            8.3, True, C_CRIMSON)], first=True)

TY = 4.05
cols = [
    ("Operating Point / Policy", 2.66, PP_ALIGN.LEFT),
    ("Cora Pool\n(527 shared neg)", 1.36, PP_ALIGN.CENTER),
    ("Citeseer Pool\n(455 shared neg)", 1.44, PP_ALIGN.CENTER),
    ("Cora HeaRT\n(500 hard/query)", 1.40, PP_ALIGN.CENTER),
    ("Citeseer HeaRT\n(500 hard/query)", 1.44, PP_ALIGN.CENTER),
]

table_rows = [
    ("Chance level (random ranking)", "1.3", "1.5", "1.4", "1.4", False, C_WHITE),
    ("Adamic–Adar (AA) alone  [0% GPU, <1 ms CPU]", "33.1  (31.9 avg)", "31.5  (29.4 avg)", "12.0  (11.9 avg)", "11.0  (10.8 avg)", False, C_WHITE),
    ("★ Frugal Hybrid  [val-picked ≤ 10% GPU cap]", "38.4 ± 0.1  (7%)", "34.7 ± 2.6  (9%)", "12.8 ± 0.2  (0.4%)", "13.8 ± 0.5  (8%)", True, C_IVORY),
    ("★ Cross-Graph Rule  [Cora → Citeseer, 25% cap]", "42.0 ± 3.9  (24%)", "44.9  (beats own 37.9)", "12.9 ± 0.1  (23%)", "13.8  (21% GPU)", True, C_WHITE),
    ("★ Half-Fanout PENCIL  [s = 2×10, all candidates]", "42.9 ± 1.9  (67%)", "51.4 ± 0.7  (63%)", "14.3 ± 0.0  (65%)", "14.2 ± 0.0  (69%)", True, C_IVORY),
    ("Official Full PENCIL  [s = 2×20, 100% GPU, 2.2 GB]", "43.7 ± 0.9  (100%)", "49.7 ± 0.2  (100%)", "13.3 ± 0.1  (100%)", "17.8 ± 0.1  (100%)", False, C_WHITE),
]

# Header band
hdr_h = 0.36
rect(RX, TY, RW, hdr_h, fill=C_NAVY)
cx = RX
for title, cw, al in cols:
    tf = tbox(cx + 0.06, TY + 0.04, cw - 0.12, hdr_h - 0.06)
    add_p(tf, [(title, 7.6, True, C_WHITE)], align=al, first=True, line_spacing=1.0)
    cx += cw

# Data rows
row_h = 0.315
for r_idx, (r0, r1, r2, r3, r4, is_hl, bg_col) in enumerate(table_rows):
    ry = TY + hdr_h + r_idx * row_h
    rect(RX, ry, RW, row_h, fill=bg_col)
    rect(RX, ry + row_h - 0.01, RW, 0.01, fill=C_RULE)
    if is_hl:
        rect(RX, ry, 0.04, row_h, fill=C_CRIMSON)

    vals = [r0, r1, r2, r3, r4]
    cx = RX
    for c_idx, ((_, cw, al), val) in enumerate(zip(cols, vals)):
        tf = tbox(cx + 0.07, ry + 0.06, cw - 0.14, row_h - 0.08)
        f_col = C_CRIMSON if (is_hl and c_idx > 0 and ("51.4" in val or "44.9" in val or "14.3" in val or "38.4" in val)) \
            else (C_INK if (is_hl or c_idx == 0) else C_SLATE)
        f_name = "Segoe UI" if c_idx == 0 else "Consolas"
        f_sz = 7.6 if c_idx == 0 else 7.5
        add_p(tf, [(val, f_sz, is_hl or (r_idx == 5 and c_idx == 4), f_col, f_name)],
              align=al, first=True, line_spacing=1.0)
        cx += cw

# Bottom heavy rule of booktabs table
table_bot = TY + hdr_h + len(table_rows) * row_h
rect(RX, table_bot, RW, 0.022, fill=C_INK)

# Footnote under table
tf = tbox(RX, table_bot + 0.04, RW, 0.22)
add_p(tf, [
    ("Protocol note: ", 7.1, True, C_SLATE),
    ("Parentheses in AA row show ties-averaged MRR (matching published HeaRT tables: 31.87, 29.37, 11.91, 10.82); degree tie-break lifts AA to 33.1 / 31.5. Peak VRAM under HeaRT drops from ", 7.1, False, C_MUTED),
    ("2,219 MB (s=2×20) → 1,347 MB (s=2×10) → 558 MB (s=2×2)", 7.1, True, C_SLATE),
    (".", 7.1, False, C_MUTED),
], first=True)

# --- Section VI: Bottom Crimson & Ivory Callout Banner (Road to Nov 26) ---
BY, BH = 6.56, 0.68
rect(RX, BY, 1.56, BH, fill=C_CRIMSON)
tf = tbox(RX + 0.08, BY + 0.13, 1.40, BH - 0.20)
add_p(tf, [("ROADMAP → NOV 26", 8.2, True, C_WHITE)], align=PP_ALIGN.CENTER, first=True, line_spacing=1.05)
add_p(tf, [("Final Report · Video · Code", 7.0, False, C_IVORY)], align=PP_ALIGN.CENTER, space_before=2, line_spacing=1.0)

rect(RX + 1.56, BY, RW - 1.56, BH, fill=C_IVORY, line=C_AMBER, line_pt=0.75)
tf = tbox(RX + 1.68, BY + 0.07, RW - 1.76, BH - 0.12)
add_p(tf, [
    ("1. Training-axis compute law: ", 7.5, True, C_INK),
    ("93% of final Cora Test MRR is reached by epoch 42 of 150 (28% of training budget), after which validation/test curves plateau.", 7.4, False, C_SLATE),
], first=True, space_after=2, line_spacing=1.04)
add_p(tf, [
    ("2. Remaining deliverables: ", 7.5, True, C_INK),
    ("Fit closed-form (k, s) selector on validation · latency + VRAM Pareto frontier · reproducible suite & 112 score dumps at ", 7.4, False, C_SLATE),
    ("github.com/Ask-812/da4022-pencil-budget", 7.4, True, C_CRIMSON, "Consolas"),
], line_spacing=1.04)

out = os.path.join(HERE, "docs", "MIDREPORT_public.pptx") if os.environ.get("PENCIL_PUBLIC") \
    else os.path.join(HERE, "MIDREPORT.pptx")
prs.save(out)
print("wrote", out)
