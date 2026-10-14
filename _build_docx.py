"""Build a submission-ready .docx of the half-page proposal."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.8)
    s.left_margin = s.right_margin = Inches(1.0)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)

def para(runs, align=None, size=11, space_after=8):
    p = doc.add_paragraph()
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    for text, bold, italic in runs:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
    return p

# --- header ---
para([("DA4022 — Course Project Proposal", True, False)],
     align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=4)
para([("How Much PENCIL Can You Afford? "
       "Compute-Allocation Laws for Link Prediction", True, False)],
     align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=10)

from team import team_line, NAMES
para([("Team: ", True, False),
      (team_line([NAMES[2], NAMES[0], NAMES[1], NAMES[3], NAMES[4]]), False, False)],
     size=10, space_after=3)
para([("Type: ", True, False),
      ("Self-proposed, extending our paper presentation — ", False, False),
      ("Plain Transformers are Surprisingly Powerful Link Predictors", False, True),
      (" (PENCIL, ICML 2026). Closest anchor: #2.", False, False)], size=10, space_after=12)

# --- body ---
para([("PENCIL matches heuristic-informed GNNs at link prediction using a plain Transformer over "
       "sampled subgraphs. Its accuracy, though, is bought with compute: Section 7 states that "
       "\u201cits GPU compute and memory scale linearly with the number of candidate links,\u201d "
       "and the authors name resource-efficient variants as future work. Expressive power "
       "likewise rises with the sampling budget, so quality is capped by what one can afford ",
       False, False),
      ("per edge", False, True),
      (". Neither the cost of a prediction, nor the accuracy that cost buys, is ever reported as "
       "a first-class result.", False, False)])

para([("We propose to supply that analysis. Production retrieval systems never score every "
       "candidate with an expensive model: a cheap stage shortlists candidates and the expensive "
       "model re-ranks the top ", False, False),
      ("k", False, True),
      (". PENCIL\u2019s authors suggest exactly this hybrid, citing ContextGNN, and never test "
       "it. We will build it \u2014 Adamic\u2013Adar as the first stage, PENCIL as the re-ranker "
       "\u2014 then treat inference as a budget-allocation problem between ", False, False),
      ("k", False, True),
      (", the number of candidates re-ranked, and ", False, False),
      ("s", False, True),
      (", the sampling budget each one receives. Under a fixed budget, is it better to re-rank "
       "many candidates shallowly or few deeply? We will sweep both on cora, citeseer and "
       "ogbl-collab, measuring accuracy against latency and peak memory, map the "
       "accuracy-versus-cost Pareto frontier, and fit a rule for choosing (", False, False),
      ("k", False, True), (", ", False, False), ("s", False, True),
      (") which we then test on a graph it was not fitted on.", False, False)])

para([("Feasibility is already established: we have reproduced the paper\u2019s cora result "
       "\u2014 test MRR ", False, False),
      ("44.00", True, False),
      (" against the published ", False, False),
      ("42.23", True, False),
      (" \u2014 in 3.2 hours on a single free Kaggle T4, where the authors used four to eight "
       "A100-class GPUs. We hold the trained checkpoint, and the ", False, False),
      ("k", False, True),
      (" axis needs no retraining, since the number of candidates re-ranked is purely an "
       "inference choice.", False, False)])

para([("As this is a self-proposed project, we would be grateful for your approval. Should "
       "self-proposed projects not be permitted, we would like this registered under ", False, True),
      ("anchor 2 (task-dependent compute allocation laws)", True, True),
      (", which this project directly addresses \u2014 it derives exactly such a law, for link "
       "prediction, on the inference budget.", False, True)], size=10, space_after=0)

import os
out = os.path.join("docs", "PROPOSAL_public.docx") if os.environ.get("PENCIL_PUBLIC") else "PROPOSAL_FINAL.docx"
doc.save(out)

import os
print("wrote", out, os.path.getsize(out), "bytes")
print("paragraphs:", len(doc.paragraphs))
words = sum(len(p.text.split()) for p in doc.paragraphs)
print("word count:", words, "-> ~%.2f page" % (words / 500))
