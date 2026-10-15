"""Build the one-slide DA4022 mid-project report.

The slide is designed as a presentation, not a compressed report page:
one claim, four directly comparable results, a compact method diagram,
and a clear statement of completed and remaining work.
"""

import csv
import json
import os
from statistics import mean, pstdev

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from team import NAMES, ROLLS


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PUBLIC = os.environ.get("PENCIL_PUBLIC") == "1"

# Warm editorial palette.
PAPER = RGBColor(0xF7, 0xF3, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x17, 0x1A, 0x1F)
SLATE = RGBColor(0x43, 0x49, 0x52)
MUTED = RGBColor(0x77, 0x7D, 0x85)
RULE = RGBColor(0xD9, 0xD4, 0xCB)
CRIMSON = RGBColor(0x9D, 0x1B, 0x32)
NAVY = RGBColor(0x20, 0x36, 0x57)
TEAL = RGBColor(0x0F, 0x76, 0x70)
AMBER = RGBColor(0xB4, 0x63, 0x18)
PALE_ROSE = RGBColor(0xF4, 0xE7, 0xE9)
PALE_TEAL = RGBColor(0xE4, 0xF0, 0xED)
PALE_AMBER = RGBColor(0xF5, 0xEA, 0xDA)
PALE_BLUE = RGBColor(0xE8, 0xED, 0xF3)


def load_metrics():
    selection = json.load(open(os.path.join(RESULTS, "selection.json"), encoding="utf-8"))
    order = ["cora_pool", "citeseer_pool", "cora_heart", "citeseer_heart"]
    labels = ["Cora\nPool", "Citeseer\nPool", "Cora\nHeaRT", "Citeseer\nHeaRT"]
    metrics = []

    for tag, label in zip(order, labels):
        rows = []
        with open(os.path.join(RESULTS, tag, "ksweep.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (
                    row["split"] == "test"
                    and row["budget"] == "d2_f10"
                    and abs(float(row["frac"]) - 1.0) < 1e-12
                ):
                    rows.append(float(row["mrr"]) * 100)

        if not rows:
            raise RuntimeError(f"Missing d2_f10 full-candidate rows for {tag}")

        selected_10 = next(r for r in selection[tag]["rows"] if abs(r["cap"] - 0.1) < 1e-12)
        metrics.append(
            {
                "tag": tag,
                "label": label,
                "aa": selection[tag]["aa_test"],
                "frugal": selected_10["test"],
                "frugal_cost": selected_10["cost"] * 100,
                "half": mean(rows),
                "half_std": pstdev(rows),
                "full": selection[tag]["full_test"],
                "full_std": selection[tag]["full_std"],
            }
        )
    return metrics


def make_comparison_figure(metrics):
    out = os.path.join(RESULTS, "fig_mid_v2.png")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#D9D4CB",
            "axes.labelcolor": "#434952",
            "xtick.color": "#777D85",
            "ytick.color": "#777D85",
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.6), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    colors = ["#C9CDD2", "#B46318", "#0F7670", "#203657"]
    names = ["AA", "≤10%", "2×10", "2×20"]

    for ax, item in zip(axes, metrics):
        values = [item["aa"], item["frugal"], item["half"], item["full"]]
        bars = ax.bar(range(4), values, color=colors, width=0.68)
        lower = max(0, min(values) - max(2.0, (max(values) - min(values)) * 0.35))
        upper = max(values) + max(2.0, (max(values) - min(values)) * 0.28)
        ax.set_ylim(lower, upper)
        ax.set_xticks(range(4), names, fontsize=7.3)
        ax.set_title(item["label"].replace("\n", " · "), fontsize=9.2, color="#171A1F", pad=7)
        ax.grid(axis="y", color="#E8E4DD", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=6.8)
        ax.tick_params(axis="x", length=0, pad=4)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (upper - lower) * 0.035,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color="#171A1F",
            )

    fig.text(0.009, 0.54, "Test MRR ×100", rotation=90, va="center", color="#777D85", fontsize=7.5)
    fig.text(
        0.5,
        0.01,
        "AA = CPU heuristic  ·  ≤10% = validation-selected hybrid  ·  2×10 = half fanout  ·  2×20 = official full PENCIL",
        ha="center",
        color="#5E646D",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.82, bottom=0.25, wspace=0.28)
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def shape(kind, x, y, w, h, fill=None, line=None, line_width=0.7, radius=False):
    if radius:
        kind = MSO_SHAPE.ROUNDED_RECTANGLE
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_width)
    if radius:
        try:
            sh.adjustments[0] = 0.08
        except Exception:
            pass
    return sh


def textbox(x, y, w, h, text="", margin=0.0, valign=MSO_ANCHOR.TOP):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    if text:
        tf.paragraphs[0].text = text
    return sh


def paragraph(
    tf,
    runs,
    *,
    first=False,
    align=PP_ALIGN.LEFT,
    line_spacing=1.0,
    space_after=0,
    space_before=0,
):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    for text, size, bold, color, font, italic in runs:
        r = p.add_run()
        r.text = text
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return p


def run(text, size, bold=False, color=INK, font="Aptos", italic=False):
    return (text, size, bold, color, font, italic)


def rule(x, y, w, color=RULE, height=0.012):
    shape(MSO_SHAPE.RECTANGLE, x, y, w, height, fill=color)


metrics = load_metrics()
chart_path = make_comparison_figure(metrics)

# Canvas.
shape(MSO_SHAPE.RECTANGLE, 0, 0, 13.333, 7.5, fill=PAPER)
shape(MSO_SHAPE.RECTANGLE, 0, 0, 13.333, 0.075, fill=CRIMSON)
shape(MSO_SHAPE.RECTANGLE, 0.18, 0.20, 0.05, 7.04, fill=CRIMSON)

# Header: report identity on the left, team on the right.
tf = textbox(0.39, 0.19, 7.75, 0.18).text_frame
paragraph(
    tf,
    [run("DA4022 MID-PROJECT REPORT  ·  SELF-PROPOSED PENCIL EXTENSION  ·  15 OCT 2026", 7.8, True, CRIMSON)],
    first=True,
)
tf = textbox(0.39, 0.39, 7.65, 0.43).text_frame
paragraph(tf, [run("How Much PENCIL Can You Afford?", 23.0, True, INK, "Georgia")], first=True)
tf = textbox(0.39, 0.79, 7.65, 0.18).text_frame
paragraph(
    tf,
    [run("Compute-allocation laws for link prediction  ·  choosing shortlist k and context budget s without retraining", 8.8, True, SLATE)],
    first=True,
)

team_parts = []
for name in NAMES:
    member = name
    if not PUBLIC:
        member += f"  {ROLLS[name]}"
    team_parts.append(member)
tf = textbox(8.05, 0.23, 4.89, 0.60).text_frame
paragraph(tf, [run("TEAM", 7.4, True, CRIMSON)], first=True, align=PP_ALIGN.RIGHT)
paragraph(
    tf,
    [run("  ·  ".join(team_parts), 6.6, False, MUTED, "Aptos")],
    align=PP_ALIGN.RIGHT,
    space_before=2,
    line_spacing=1.05,
)
rule(0.39, 1.02, 12.55)

# Three-column editorial grid.
LX, LW = 0.39, 3.00
MX, MW = 3.55, 4.31
RX, RW = 8.02, 4.92
shape(MSO_SHAPE.RECTANGLE, 3.46, 1.20, 0.012, 5.91, fill=RULE)
shape(MSO_SHAPE.RECTANGLE, 7.93, 1.20, 0.012, 5.91, fill=RULE)

# -----------------------------------------------------------------------------
# COLUMN 1 — METHOD, REPRODUCTION, AUDIT, ROADMAP
# -----------------------------------------------------------------------------
tf = textbox(LX, 1.15, LW, 0.18).text_frame
paragraph(tf, [run("WHAT WE ARE DOING", 8.2, True, CRIMSON)], first=True)

shape(MSO_SHAPE.ROUNDED_RECTANGLE, LX, 1.40, LW, 1.65, fill=WHITE, line=RULE, radius=True)
shape(MSO_SHAPE.RECTANGLE, LX, 1.40, 0.045, 1.65, fill=NAVY)
tf = textbox(LX + 0.13, 1.50, LW - 0.24, 0.68).text_frame
paragraph(
    tf,
    [
        run("We rank every candidate link with Adamic–Adar, run PENCIL only on the top-k candidates, and vary PENCIL's sampled subgraph budget s. ", 7.5, False, SLATE),
        run("The (k, s) sweep measures how much accuracy remains at each GPU budget. The model is not retrained.", 7.5, True, INK),
    ],
    first=True,
    line_spacing=1.05,
)

steps = [
    ("1", "AA rank", "CPU <1 ms", PALE_AMBER, AMBER),
    ("2", "top-k", "shortlist 0–100%", PALE_ROSE, CRIMSON),
    ("3", "PENCIL", "re-rank with s", PALE_BLUE, NAVY),
]
for i, (num, title, body, bg, accent) in enumerate(steps):
    x = LX + 0.12 + i * 0.94
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.32, 0.82, 0.55, fill=bg, line=RULE, radius=True)
    tf = textbox(x + 0.05, 2.38, 0.72, 0.16).text_frame
    paragraph(tf, [run(f"{num}  {title}", 7.3, True, accent)], first=True, align=PP_ALIGN.CENTER)
    tf = textbox(x + 0.03, 2.62, 0.76, 0.15).text_frame
    paragraph(tf, [run(body, 5.9, False, SLATE)], first=True, align=PP_ALIGN.CENTER)
    if i < 2:
        tf = textbox(x + 0.82, 2.48, 0.12, 0.18).text_frame
        paragraph(tf, [run("→", 8.5, True, MUTED)], first=True, align=PP_ALIGN.CENTER)

tf = textbox(LX, 3.20, LW, 0.18).text_frame
paragraph(tf, [run("FREE-GPU REPRODUCTION  ·  PAPER USED 4–8× A100", 7.7, True, CRIMSON)], first=True)
kpis = [
    ("44.00", "Cora test", "paper 42.23", NAVY),
    ("50.46", "Citeseer test", "paper 47.51", TEAL),
    ("112", "score dumps", "4 regimes", CRIMSON),
]
for i, (value, title, sub, accent) in enumerate(kpis):
    x = LX + i * 1.01
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, 3.44, 0.92, 0.82, fill=WHITE, line=RULE, radius=True)
    shape(MSO_SHAPE.RECTANGLE, x, 3.44, 0.92, 0.04, fill=accent)
    tf = textbox(x + 0.06, 3.56, 0.80, 0.25).text_frame
    paragraph(tf, [run(value, 14.5, True, accent, "Georgia")], first=True, align=PP_ALIGN.CENTER)
    tf = textbox(x + 0.04, 3.83, 0.84, 0.14).text_frame
    paragraph(tf, [run(title, 6.8, True, INK)], first=True, align=PP_ALIGN.CENTER)
    tf = textbox(x + 0.04, 4.02, 0.84, 0.13).text_frame
    paragraph(tf, [run(sub, 6.2, False, MUTED, "Consolas")], first=True, align=PP_ALIGN.CENTER)

tf = textbox(LX, 4.45, LW, 0.18).text_frame
paragraph(tf, [run("EXPERIMENT SETUP", 8.2, True, CRIMSON)], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, LX, 4.69, LW, 1.38, fill=WHITE, line=RULE, radius=True)
audit_rows = [
    ("Data", "Cora and Citeseer"),
    ("Tests", "shared negative pool and HeaRT hard negatives"),
    ("Selection", "choose (k, s) on validation under fixed compute caps"),
    ("Checks", "corrected score-precision and shortlist tie bias"),
]
for i, (head, body) in enumerate(audit_rows):
    y = 4.79 + i * 0.30
    if i:
        rule(LX + 0.12, y - 0.045, LW - 0.24, color=RULE, height=0.008)
    tf = textbox(LX + 0.13, y, 0.66, 0.16).text_frame
    paragraph(tf, [run(head, 7.0, True, NAVY)], first=True)
    tf = textbox(LX + 0.79, y, 2.05, 0.19).text_frame
    paragraph(tf, [run(body, 6.55, False, SLATE)], first=True)

tf = textbox(LX, 6.27, LW, 0.18).text_frame
paragraph(tf, [run("WHAT WE MEASURE", 8.2, True, CRIMSON)], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, LX, 6.51, LW, 0.49, fill=PALE_AMBER, line=RGBColor(0xDC, 0xBE, 0x99), radius=True)
tf = textbox(LX + 0.13, 6.63, LW - 0.26, 0.22).text_frame
paragraph(
    tf,
    [run("Test MRR ×100  ·  measured T4 GPU time  ·  peak VRAM  ·  three random seeds per budget", 6.8, True, SLATE)],
    first=True,
    align=PP_ALIGN.CENTER,
)

# -----------------------------------------------------------------------------
# COLUMN 2 — FOUR DETAILED EMPIRICAL FINDINGS
# -----------------------------------------------------------------------------
tf = textbox(MX, 1.15, MW, 0.18).text_frame
paragraph(tf, [run("RESULTS FROM THE BUDGET SWEEP", 8.2, True, CRIMSON)], first=True)

findings = [
    (
        "01",
        "Half fanout is enough in 3 of 4 regimes",
        "At s=2×10 (~65% GPU), PENCIL matches Cora Pool (42.9 vs 43.7) and beats full s=2×20 on Citeseer Pool (51.4 vs 49.7) and Cora HeaRT (14.3 vs 13.3). Citeseer HeaRT is the exception: 14.2 vs 17.8.",
        NAVY,
        PALE_BLUE,
    ),
    (
        "02",
        "A ≤10% GPU hybrid beats heuristics everywhere",
        "Validation-selected (k, s) improves over Adamic–Adar in all four tests: 33.1→38.4 on Cora Pool, 31.5→34.7 on Citeseer Pool, 12.0→12.8 on Cora HeaRT, and 11.0→13.8 on Citeseer HeaRT.",
        TEAL,
        PALE_TEAL,
    ),
    (
        "03",
        "PENCIL pays off where graph structure is blind",
        "Re-ranking only the top 10% changes almost nothing because AA already ranks easy links highly. The useful compute lies in the zero-common-neighbour tail: 57.3% of true Cora test links have zero shared neighbours.",
        AMBER,
        PALE_AMBER,
    ),
    (
        "04",
        "Cross-graph rules can beat noisy local selection",
        "Applying Cora's validation-chosen rule unchanged to Citeseer beats Citeseer's own validation pick at every cap ≤50%; at the 25% cap, transferred MRR is 44.9 versus 37.9 from local selection.",
        CRIMSON,
        PALE_ROSE,
    ),
]

card_y = [1.40, 2.68, 3.96, 5.24]
for (num, title, body, accent, tint), y in zip(findings, card_y):
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, MX, y, MW, 1.12, fill=WHITE, line=RULE, radius=True)
    shape(MSO_SHAPE.RECTANGLE, MX, y, 0.045, 1.12, fill=accent)
    shape(MSO_SHAPE.OVAL, MX + 0.15, y + 0.18, 0.42, 0.42, fill=tint, line=accent, line_width=0.8)
    tf = textbox(MX + 0.15, y + 0.25, 0.42, 0.18, valign=MSO_ANCHOR.MIDDLE).text_frame
    paragraph(tf, [run(num, 8.0, True, accent, "Georgia")], first=True, align=PP_ALIGN.CENTER)
    tf = textbox(MX + 0.69, y + 0.13, MW - 0.84, 0.25).text_frame
    paragraph(tf, [run(title, 9.2, True, INK, "Georgia")], first=True)
    tf = textbox(MX + 0.69, y + 0.43, MW - 0.84, 0.57).text_frame
    paragraph(tf, [run(body, 7.05, False, SLATE)], first=True, line_spacing=1.04)

shape(MSO_SHAPE.ROUNDED_RECTANGLE, MX, 6.55, MW, 0.45, fill=NAVY, radius=True)
tf = textbox(MX + 0.16, 6.65, MW - 0.32, 0.23, valign=MSO_ANCHOR.MIDDLE).text_frame
paragraph(
    tf,
    [run("Main result: ", 7.4, True, PALE_BLUE), run("reducing fanout is usually safer than shortening the candidate list.", 7.6, True, WHITE)],
    first=True,
    align=PP_ALIGN.CENTER,
)

# -----------------------------------------------------------------------------
# COLUMN 3 — FIGURES, BENCHMARK MATRIX, TRANSFER + RESOURCE EVIDENCE
# -----------------------------------------------------------------------------
tf = textbox(RX, 1.15, RW, 0.18).text_frame
paragraph(tf, [run("BUDGET SWEEPS", 8.2, True, CRIMSON)], first=True)

shape(MSO_SHAPE.ROUNDED_RECTANGLE, RX, 1.40, RW, 1.92, fill=WHITE, line=RULE, radius=True)
slide.shapes.add_picture(
    os.path.join(RESULTS, "fig_mid.png"),
    Inches(RX + 0.08),
    Inches(1.50),
    width=Inches(RW - 0.16),
    height=Inches(1.44),
)
tf = textbox(RX + 0.15, 3.00, RW - 0.30, 0.19).text_frame
paragraph(
    tf,
    [run("Shortlist saturation  ·  compute Pareto frontiers  ·  HeaRT hard-negative stress test", 6.5, False, MUTED)],
    first=True,
    align=PP_ALIGN.CENTER,
)

tf = textbox(RX, 3.49, RW, 0.18).text_frame
paragraph(tf, [run("VALIDATION-SELECTED OPERATING POINTS  ·  TEST MRR ×100", 8.2, True, CRIMSON)], first=True)

table_x, table_y = RX, 3.74
col_widths = [1.34, 0.88, 0.88, 0.90, 0.92]
headers = ["Policy", "Cora\nPool", "Citeseer\nPool", "Cora\nHeaRT", "Citeseer\nHeaRT"]
rows = [
    ("Chance", "1.3", "1.5", "1.4", "1.4"),
    ("AA · 0% GPU", "33.1", "31.5", "12.0", "11.0"),
    ("Hybrid · ≤10%", "38.4 · 7%", "34.7 · 9%", "12.8 · 0.4%", "13.8 · 8%"),
    ("Half · s=2×10", "42.9 · 67%", "51.4 · 63%", "14.3 · 65%", "14.2 · 69%"),
    ("Full · s=2×20", "43.7", "49.7", "13.3", "17.8"),
]
header_h, row_h = 0.40, 0.315
shape(MSO_SHAPE.RECTANGLE, table_x, table_y, RW, header_h, fill=NAVY)
cx = table_x
for i, (head, width) in enumerate(zip(headers, col_widths)):
    tf = textbox(cx + 0.04, table_y + 0.07, width - 0.08, 0.27).text_frame
    paragraph(
        tf,
        [run(head, 6.7, True, WHITE)],
        first=True,
        align=PP_ALIGN.LEFT if i == 0 else PP_ALIGN.CENTER,
        line_spacing=0.95,
    )
    cx += width

for r_idx, values in enumerate(rows):
    y = table_y + header_h + r_idx * row_h
    bg = PALE_AMBER if r_idx == 2 else (PALE_TEAL if r_idx == 3 else WHITE)
    shape(MSO_SHAPE.RECTANGLE, table_x, y, RW, row_h, fill=bg)
    rule(table_x, y + row_h - 0.008, RW, color=RULE, height=0.008)
    cx = table_x
    for c_idx, (value, width) in enumerate(zip(values, col_widths)):
        color = INK
        bold = c_idx == 0 or r_idx in (2, 3)
        if r_idx == 3 and c_idx in (2, 3):
            color = TEAL
        if r_idx == 2 and c_idx in (1, 4):
            color = AMBER
        tf = textbox(cx + 0.05, y + 0.075, width - 0.10, 0.16, valign=MSO_ANCHOR.MIDDLE).text_frame
        paragraph(
            tf,
            [run(value, 6.6, bold, color, "Aptos" if c_idx == 0 else "Consolas")],
            first=True,
            align=PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER,
        )
        cx += width

table_bottom = table_y + header_h + len(rows) * row_h
shape(MSO_SHAPE.RECTANGLE, table_x, table_bottom, RW, 0.018, fill=INK)

tf = textbox(RX, 5.87, RW, 0.18).text_frame
paragraph(tf, [run("ADDITIONAL RESULTS", 8.2, True, CRIMSON)], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, RX, 6.10, RW, 0.90, fill=WHITE, line=RULE, radius=True)
shape(MSO_SHAPE.RECTANGLE, RX, 6.10, 0.045, 0.90, fill=TEAL)
tf = textbox(RX + 0.15, 6.20, 2.30, 0.66).text_frame
paragraph(
    tf,
    [run("Cora → Citeseer", 7.3, True, TEAL), run("  At the 25% cap, the transferred rule reaches 44.9 MRR, beating Citeseer's own noisy validation pick (37.9).", 6.7, False, SLATE)],
    first=True,
    line_spacing=1.03,
)
shape(MSO_SHAPE.RECTANGLE, RX + 2.53, 6.22, 0.010, 0.55, fill=RULE)
tf = textbox(RX + 2.69, 6.20, 2.04, 0.66).text_frame
paragraph(
    tf,
    [run("Peak VRAM · Cora HeaRT", 7.3, True, NAVY), run("  2,219 MB (2×20) → 1,347 MB (2×10) → 558 MB (2×2).", 6.7, False, SLATE)],
    first=True,
    line_spacing=1.03,
)

# Footer.
rule(0.39, 7.18, 12.55)
tf = textbox(0.39, 7.25, 12.55, 0.14).text_frame
paragraph(
    tf,
    [run("All operating points are selected on validation data; test data are used only for the reported results.", 6.5, True, MUTED)],
    first=True,
    align=PP_ALIGN.CENTER,
)

out = os.path.join(HERE, "docs", "MIDREPORT_public.pptx") if PUBLIC else os.path.join(HERE, "MIDREPORT.pptx")
prs.save(out)
print(f"wrote {out}")
