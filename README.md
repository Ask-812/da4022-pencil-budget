# How Much PENCIL Can You Afford?

**Compute-allocation for Transformer link prediction: a DA4022 course project.**
Team: Amit Kumar · Ashmit Satpute · Arnav Kshirsagar · Pranav Shalya · Atharv Wankhede

[PENCIL](https://github.com/quang-truong/pencil) (*Plain Transformers are Surprisingly Powerful Link
Predictors*, ICML 2026) scores every candidate link by running a Transformer over a sampled subgraph,
so its cost grows linearly with the number of candidates. The authors suggest, but never test, a
retrieve-then-rerank hybrid (citing ContextGNN, ICLR 2025). We build it and measure accuracy against compute:

```
all candidates --> Adamic-Adar (cheap, CPU) --> top k --> PENCIL with sampling budget s (GPU) --> ranking
```

Two inference-time knobs, no retraining: **k** (how many candidates PENCIL re-ranks) and
**s** (hops x neighbours sampled per candidate; official = 2 x 20).

## Results

| | Cora, pool | Citeseer, pool | Cora, HeaRT | Citeseer, HeaRT |
|---|---|---|---|---|
| Adamic-Adar alone (0% GPU) | 33.1 | 31.5 | 12.0 | 11.0 |
| Cheap: validation-picked, <= 10% compute | 38.4 (7%) | 34.7 (9%) | 12.8 (0.4%) | 13.8 (8%) |
| PENCIL, s = 2x10 (half the neighbours) | 42.9 (67%) | 51.4 (63%) | 14.3 (65%) | 14.2 (69%) |
| PENCIL, official s = 2x20 | 43.7 ± 0.9 (100%) | 49.7 ± 0.2 (100%) | 13.3 ± 0.1 (100%) | 17.8 ± 0.1 (100%) |

*Test MRR x100. Pool = planetoid protocol (each positive against all test negatives).
HeaRT = Li et al., NeurIPS 2023 (each positive against its own 500 hard negatives).
Compute = PENCIL forward time per candidate on a T4, relative to the official setting.
"Cheap" = (k, s) chosen on **validation** under a 10% compute cap; test touched once.*

**Findings**
1. **The official sampling budget usually over-reads.** s = 2x10 is within 1 MRR point of s = 2x20,
   or better, in 3 of 4 settings at about 65% of the compute. The exception is the hardest setting
   (Citeseer, HeaRT: 14.2 vs 17.8), so s should be tuned per task, not copied from the paper.
2. **Re-ranking the top of the shortlist is wasted.** AA's top candidates are already right; PENCIL
   earns its cost on pairs with no common neighbour (57% of Cora's true test links).
3. **Under HeaRT, more re-ranking can hurt.** Re-ranking the top 1% per query captures most of
   PENCIL's gain on Cora at <1% compute; weak re-rankers (1 hop) fall below AA alone. PENCIL's
   HeaRT edge over AA is small on Cora (13.3 vs 12.0) and large on Citeseer (17.8 vs 11.0; paper 16.80).
4. **Budget rules transfer under the pool protocol, only partly under HeaRT.** (k, s) chosen on Cora's
   validation set match or beat Citeseer's own (tiny, noisy) validation choice at every pool budget
   (e.g. 44.9 vs 37.9 at 25%). Under HeaRT they hold at small budgets but cost 3.6 points at full budget,
   because Cora prefers s = 2x10, the setting that fails on Citeseer-HeaRT.

Per-setting figures: `results/<dataset>_<protocol>/fig.png`. Selection and transfer tables:
`results/selection.json` (printed by `python select_rule.py`). Full write-up: [`RESULTS.md`](RESULTS.md).

**Reproduction of the paper on free hardware:** Cora 44.00 test MRR (paper 42.23), 3.2 h on one T4;
Citeseer 50.46 (paper 47.51), 1 h on two T4s. The paper used 4-8 A100-class GPUs.

**Checks.** Our AA reproduces the HeaRT benchmark exactly (Cora 31.87 pool / 11.91 HeaRT,
Citeseer 29.37 pool / 10.82 HeaRT). Our metric matches the repo evaluator (max |diff| ~1e-7). `analysis.py` asserts
k = all equals full PENCIL and k = 0 equals AA, and breaks retriever ties at random.

## Reproduce

Everything heavy runs on free Kaggle T4s; analysis runs on a laptop CPU.

```bash
pip install torch numpy scipy matplotlib python-pptx python-docx kaggle remotezip pymupdf

# 1. data: pull only the needed files from the authors' 7.5 GB archive (HTTP range requests)
python dl/pull_cora.py ; python dl/pull_planetoid.py      # -> Kaggle datasets pencil-cora / pencil-planetoid

# 2. train (Kaggle).  args: dataset  kernel-slug  data-source  num-gpus
python _build_baseline.py                                              # cora, 1 x T4, ~3.2 h
python _build_baseline.py citeseer ask812/pencil-citeseer ask812/pencil-planetoid 2
cd kernel_train_citeseer && kaggle kernels push -p . --accelerator NvidiaTeslaT4

# 3. score every candidate once per s (Kaggle).  args: dataset  heart(0|1)
python _build_eval.py cora 1 && cd kernel_eval_heart && kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels output ask812/pencil-budget-eval-heart -p evals_dl/cora_heart   # unzip to evals/cora_heart/

# 4. analysis (CPU, seconds)
python analysis.py cora heart        # -> results/cora_heart/{ksweep.csv,summary.json}
python make_figures.py cora_heart
python select_rule.py                # validation-selected rule + transfer test
```

The per-candidate score dumps are attached to the [v1.0 release](https://github.com/Ask-812/da4022-pencil-budget/releases/tag/v1.0):
unzip each `evals_*.zip` into `evals/`, run step 1 for the split files, and step 4 runs on a CPU without Kaggle.

**Three things the official code needs on a T4** (patched automatically by the kernels):
`transformers==4.46.2` (5.x returns a tensor from `BertLayer`, breaking PENCIL's multiplicative residual);
batch 64 with accumulation scaled to keep the official effective batch; `WANDB_MODE=disabled` + `torchrun`.

## Layout

| path | what |
|---|---|
| `_build_baseline.py`, `_build_eval.py` | generate the Kaggle training / scoring notebooks (`kernel_*/`) |
| `analysis.py` | retrieve-then-rerank k-sweep for both protocols, with self-checks |
| `select_rule.py` | validation-selected (k, s) per compute cap, cross-graph transfer |
| `make_figures.py` | per-setting accuracy-vs-compute figures |
| `_build_slide.py`, `_build_docx.py` | mid-project slide and proposal (`docs/`) |
| `results/` | sweep tables, summaries, figures |
| `dl/` | partial-download scripts for the authors' data archive |

## Credits

PENCIL code and data: Truong et al. (ICML 2026), <https://github.com/quang-truong/pencil>; not
redistributed here, cloned at run time. HeaRT negatives: Li et al. (NeurIPS 2023).
Our code: MIT licence (see `LICENSE`).
