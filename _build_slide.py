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
    fig, axes = plt.subplots(1, 4, figsize=(8.15, 1.86), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")
    colors = ["#C9CDD2", "#B46318", "#0F7670", "#203657"]
    names = ["AA", "≤10%", "2×10", "2×20"]

    for ax, item in zip(axes, metrics):
        values = [item["aa"], item["frugal"], item["half"], item["full"]]
        bars = ax.bar(range(4), values, color=colors, width=0.68)
        lower = max(0, min(values) - max(2.0, (max(values) - min(values)) * 0.35))
        upper = max(values) + max(2.0, (max(values) - min(values)) * 0.28)
        ax.set_ylim(lower, upper)
        ax.set_xticks(range(4), names, fontsize=7.0)
        ax.set_title(item["label"].replace("\n", " · "), fontsize=8.5, color="#171A1F", pad=6)
        ax.grid(axis="y", color="#E8E4DD", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=6.5)
        ax.tick_params(axis="x", length=0, pad=4)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (upper - lower) * 0.035,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color="#171A1F",
            )

    fig.text(0.006, 0.54, "Test MRR ×100", rotation=90, va="center", color="#777D85", fontsize=7.0)
    fig.text(
        0.5,
        0.01,
        "AA = CPU heuristic  ·  ≤10% = validation-selected hybrid  ·  2×10 = half fanout  ·  2×20 = official full PENCIL",
        ha="center",
        color="#5E646D",
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.050, right=0.995, top=0.80, bottom=0.27, wspace=0.30)
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
shape(MSO_SHAPE.RECTANGLE, 0, 0, 13.333, 0.07, fill=CRIMSON)

# Header.
tf = textbox(0.42, 0.18, 7.9, 0.18).text_frame
paragraph(
    tf,
    [run("DA4022 mid-project report  ·  PENCIL extension  ·  15 October 2026", 8.2, True, CRIMSON, "Segoe UI")],
    first=True,
)
tf = textbox(0.42, 0.39, 8.2, 0.45).text_frame
paragraph(tf, [run("How Much PENCIL Can You Afford?", 24.5, True, INK, "Georgia")], first=True)
tf = textbox(0.42, 0.82, 8.2, 0.20).text_frame
paragraph(
    tf,
    [run("Compute allocation for link prediction through candidate shortlisting and smaller sampled subgraphs", 10.0, False, SLATE, "Segoe UI")],
    first=True,
)

team_parts = []
for name in NAMES:
    member = name
    if not PUBLIC:
        member += f" ({ROLLS[name]})"
    team_parts.append(member)
tf = textbox(8.65, 0.26, 4.25, 0.62).text_frame
paragraph(
    tf,
    [run("  ·  ".join(team_parts), 7.1, False, MUTED, "Segoe UI")],
    first=True,
    align=PP_ALIGN.RIGHT,
    line_spacing=1.08,
)
rule(0.42, 1.07, 12.48)

# Upper-left: method and experiment.
LX, LW = 0.42, 4.02
RX, RW = 4.68, 8.22

tf = textbox(LX, 1.22, LW, 0.25).text_frame
paragraph(tf, [run("What we are testing", 12.0, True, INK, "Georgia")], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, LX, 1.53, LW, 1.34, fill=WHITE, line=RULE, radius=True)
shape(MSO_SHAPE.RECTANGLE, LX, 1.53, 0.045, 1.34, fill=CRIMSON)
tf = textbox(LX + 0.14, 1.64, LW - 0.27, 0.68).text_frame
paragraph(
    tf,
    [
        run("We rank candidate links with Adamic–Adar, run PENCIL only on the top-k candidates, and reduce the sampled subgraph budget s. ", 9.2, False, SLATE, "Segoe UI"),
        run("Sweeping (k, s) measures the accuracy retained at each GPU budget without retraining.", 9.2, True, INK, "Segoe UI"),
    ],
    first=True,
    line_spacing=1.06,
)

steps = [
    ("1", "AA ranking", "CPU", PALE_AMBER, AMBER),
    ("2", "Top-k shortlist", "candidate count", PALE_ROSE, CRIMSON),
    ("3", "PENCIL", "context budget s", PALE_BLUE, NAVY),
]
for i, (num, title, sub, bg, accent) in enumerate(steps):
    x = LX + 0.14 + i * 1.27
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.39, 1.14, 0.36, fill=bg, line=RULE, radius=True)
    tf = textbox(x + 0.06, 2.425, 1.02, 0.26, valign=MSO_ANCHOR.MIDDLE).text_frame
    paragraph(
        tf,
        [run(f"{num}  {title}", 7.7, True, accent, "Segoe UI"), run(f"  ·  {sub}", 7.3, False, SLATE, "Segoe UI")],
        first=True,
        align=PP_ALIGN.CENTER,
    )
    if i < 2:
        tf = textbox(x + 1.14, 2.47, 0.13, 0.16).text_frame
        paragraph(tf, [run("→", 9.0, True, MUTED, "Segoe UI")], first=True, align=PP_ALIGN.CENTER)

tf = textbox(LX, 3.02, LW, 0.25).text_frame
paragraph(tf, [run("Experiment", 12.0, True, INK, "Georgia")], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, LX, 3.34, LW, 1.00, fill=WHITE, line=RULE, radius=True)
experiment_lines = [
    ("Data", "Cora and Citeseer"),
    ("Evaluation", "shared negative pool and HeaRT hard negatives"),
    ("Selection", "(k, s) chosen on validation data under fixed compute caps"),
    ("Recorded", "test MRR, measured T4 time, peak VRAM, three seeds"),
]
for i, (label, value) in enumerate(experiment_lines):
    y = 3.45 + i * 0.21
    tf = textbox(LX + 0.16, y, 0.78, 0.17).text_frame
    paragraph(tf, [run(label, 8.1, True, NAVY, "Segoe UI")], first=True)
    tf = textbox(LX + 0.94, y, 2.90, 0.17).text_frame
    paragraph(tf, [run(value, 8.1, False, SLATE, "Segoe UI")], first=True)

tf = textbox(LX, 4.52, LW, 0.25).text_frame
paragraph(tf, [run("Reproduction check", 12.0, True, INK, "Georgia")], first=True)
kpis = [
    ("44.00", "Cora test", "paper 42.23", NAVY),
    ("50.46", "Citeseer test", "paper 47.51", TEAL),
    ("112", "score dumps", "four regimes", CRIMSON),
]
for i, (value, title, sub, accent) in enumerate(kpis):
    x = LX + i * 1.36
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, 4.83, 1.23, 0.52, fill=WHITE, line=RULE, radius=True)
    tf = textbox(x + 0.05, 4.91, 0.62, 0.22).text_frame
    paragraph(tf, [run(value, 12.2, True, accent, "Georgia")], first=True, align=PP_ALIGN.CENTER)
    tf = textbox(x + 0.69, 4.86, 0.47, 0.48).text_frame
    paragraph(
        tf,
        [run(title, 7.6, True, INK, "Segoe UI"), run("\n" + sub, 7.0, False, MUTED, "Segoe UI")],
        first=True,
        align=PP_ALIGN.LEFT,
        line_spacing=0.95,
    )

# Upper-right: readable chart and result matrix.
tf = textbox(RX, 1.22, RW, 0.25).text_frame
paragraph(tf, [run("Results", 12.0, True, INK, "Georgia")], first=True)
shape(MSO_SHAPE.ROUNDED_RECTANGLE, RX, 1.53, RW, 1.88, fill=WHITE, line=RULE, radius=True)
slide.shapes.add_picture(
    chart_path,
    Inches(RX + 0.08),
    Inches(1.61),
    width=Inches(RW - 0.16),
    height=Inches(1.72),
)

tf = textbox(RX, 3.58, RW, 0.25).text_frame
paragraph(tf, [run("Test MRR at selected budgets", 12.0, True, INK, "Georgia")], first=True)
table_x, table_y = RX, 3.89
col_widths = [1.78, 1.58, 1.62, 1.60, 1.64]
headers = ["Policy", "Cora Pool", "Citeseer Pool", "Cora HeaRT", "Citeseer HeaRT"]
rows = [
    ("Chance", "1.3", "1.5", "1.4", "1.4"),
    ("Adamic–Adar · 0%", "33.1", "31.5", "12.0", "11.0"),
    ("Hybrid · ≤10%", "38.4 · 7%", "34.7 · 9%", "12.8 · 0.4%", "13.8 · 8%"),
    ("PENCIL · s=2×10", "42.9 · 67%", "51.4 · 63%", "14.3 · 65%", "14.2 · 69%"),
    ("PENCIL · s=2×20", "43.7", "49.7", "13.3", "17.8"),
]
header_h, row_h = 0.30, 0.245
shape(MSO_SHAPE.RECTANGLE, table_x, table_y, RW, header_h, fill=NAVY)
cx = table_x
for i, (head, width) in enumerate(zip(headers, col_widths)):
    tf = textbox(cx + 0.05, table_y + 0.075, width - 0.10, 0.16).text_frame
    paragraph(
        tf,
        [run(head, 7.5, True, WHITE, "Segoe UI")],
        first=True,
        align=PP_ALIGN.LEFT if i == 0 else PP_ALIGN.CENTER,
    )
    cx += width
for r_idx, values in enumerate(rows):
    y = table_y + header_h + r_idx * row_h
    bg = PALE_AMBER if r_idx == 2 else (PALE_TEAL if r_idx == 3 else WHITE)
    shape(MSO_SHAPE.RECTANGLE, table_x, y, RW, row_h, fill=bg)
    rule(table_x, y + row_h - 0.008, RW, color=RULE, height=0.008)
    cx = table_x
    for c_idx, (value, width) in enumerate(zip(values, col_widths)):
        color = TEAL if r_idx == 3 and c_idx in (2, 3) else INK
        if r_idx == 2 and c_idx in (1, 4):
            color = AMBER
        tf = textbox(cx + 0.05, y + 0.055, width - 0.10, 0.15).text_frame
        paragraph(
            tf,
            [run(value, 7.6, c_idx == 0 or r_idx in (2, 3), color, "Segoe UI")],
            first=True,
            align=PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER,
        )
        cx += width
shape(MSO_SHAPE.RECTANGLE, table_x, table_y + header_h + len(rows) * row_h, RW, 0.018, fill=INK)

# Bottom: four findings, each with readable text.
tf = textbox(0.42, 5.45, 12.48, 0.25).text_frame
paragraph(tf, [run("What the sweep shows", 12.0, True, INK, "Georgia")], first=True)

findings = [
    (
        "01",
        "Half fanout is usually enough",
        "At s=2×10, PENCIL matches Cora Pool and beats full s=2×20 on Citeseer Pool and Cora HeaRT. Citeseer HeaRT is the only case that clearly needs full context.",
        NAVY,
    ),
    (
        "02",
        "The cheap hybrid helps everywhere",
        "A validation-selected hybrid using at most 10% of full GPU compute improves over Adamic–Adar in all four test settings.",
        TEAL,
    ),
    (
        "03",
        "Compute matters in the difficult tail",
        "Re-ranking the top 10% adds little. PENCIL is useful for links that graph heuristics cannot separate; 57.3% of true Cora test links have zero common neighbours.",
        AMBER,
    ),
    (
        "04",
        "The rule transfers across graphs",
        "At a 25% cap, the rule selected on Cora reaches 44.9 MRR on Citeseer, compared with 37.9 from Citeseer's noisier local validation choice.",
        CRIMSON,
    ),
]
gap = 0.12
card_w = (12.48 - 3 * gap) / 4
for i, (num, title, body, accent) in enumerate(findings):
    x = 0.42 + i * (card_w + gap)
    shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, 5.76, card_w, 1.32, fill=WHITE, line=RULE, radius=True)
    shape(MSO_SHAPE.RECTANGLE, x, 5.76, card_w, 0.045, fill=accent)
    tf = textbox(x + 0.15, 5.90, 0.34, 0.24).text_frame
    paragraph(tf, [run(num, 13.0, True, accent, "Georgia")], first=True)
    tf = textbox(x + 0.52, 5.88, card_w - 0.67, 0.36).text_frame
    paragraph(tf, [run(title, 10.0, True, INK, "Georgia")], first=True, line_spacing=1.0)
    tf = textbox(x + 0.13, 6.27, card_w - 0.26, 0.66).text_frame
    paragraph(tf, [run(body, 8.4, False, SLATE, "Segoe UI")], first=True, line_spacing=1.06)

# Footer: method note only.
rule(0.42, 7.23, 12.48)
tf = textbox(0.42, 7.30, 12.48, 0.14).text_frame
paragraph(
    tf,
    [run("Operating points are selected on validation data; test data are used only for the reported results.", 7.3, False, MUTED, "Segoe UI")],
    first=True,
    align=PP_ALIGN.CENTER,
)

out = os.path.join(HERE, "docs", "MIDREPORT_public.pptx") if PUBLIC else os.path.join(HERE, "MIDREPORT.pptx")
prs.save(out)
print(f"wrote {out}")
