# RESULT — PENCIL reproduces on a single Kaggle T4

**Date:** 29 Aug 2026 · **Verdict: PASS — full baseline reproduced**

> **Mid-project (11 Oct 2026):** budget sweep done — see the section "Retrieve-then-rerank budget
> study" at the bottom. Slide: `MIDREPORT.pptx` / `MIDREPORT.pdf`.

Run end-to-end on Kaggle. These are measured numbers, not estimates.

---

# ★ HEADLINE: full 150-epoch cora baseline reproduced

**Kernel:** `ask812/pencil-baseline` · exit code 0 · **189.6 min (~3.2 h) on ONE Tesla T4**

| | test MRR |
|---|---|
| **Paper** (PENCIL w/o features, cora) | **42.23** |
| **Our reproduction** (1 × free T4) | **44.00** |

We match — in fact slightly exceed — the published number, on a single free GPU. The paper's
results used **4–8 × A100/A5000/A6000**. Our config used `use_features: False`, the same setting
as the paper's 42.23.

Full best metrics (validation-selected):

| | hits@1 | hits@10 | hits@50 | hits@100 | MRR |
|---|---|---|---|---|---|
| valid | 0.5095 | 0.7110 | 0.9049 | 0.9278 | 0.5941 |
| test | 0.3928 | 0.5522 | 0.7856 | 0.8729 | **0.4400** |

Artefacts saved: `best_model_checkpoint.pth` (316 MB), `metrics.jsonl` (75 eval points),
`best_metrics.json`, `config.yaml`.

### ⚡ A free finding, straight out of the baseline

Test MRR over training: ep22 `0.251` → ep32 `0.356` → **ep42 `0.407`** → ep62 `0.418`
→ ep102 `0.422` → ep142 `0.414`; best `0.440`.

**~93% of final performance is reached by epoch 42 of 150** — i.e. at ~28% of the training
budget, after which the curve is flat. That is a compute-allocation result about the *training*
axis, obtained before the project formally starts, and it fits the proposal's thesis directly.

Also note the **valid 0.594 vs test 0.440 gap** — worth a sentence in the report.

---

## Spike (how we got here)

## Outcome (spike, 3 epochs)

PENCIL trained on `cora` for 3 epochs on **one Tesla T4**, exit code 0, metrics written.

| Epoch | valid MRR | valid hits@100 | test MRR |
|---|---|---|---|
| 1 | 0.0756 | 0.517 | 0.0780 |
| 2 | 0.0768 | 0.624 | 0.0734 |
| 3 | **0.0866** | **0.646** | 0.0669 |

Loss fell 0.90 → 0.68 within epoch 3. MRR and hits@100 both rising — the model is learning,
not just running. These are 3-of-150 epochs, so the absolute values are meaningless; the
*pipeline* is what was being tested.

**Model size: 28,184,577 params** — consistent with the ~27.3M we quoted in the presentation.

## Measured environment (Kaggle, Aug 2026)

| | |
|---|---|
| GPU | Tesla T4, 14.6 GB, **sm_75**, 2 available |
| torch | **2.10.0+cu128** |
| Python | 3.12.13 |
| CPUs | 4 |

## The three blockers, and their fixes

Each was found by running, not guessing. All three are now solved.

### 1. `transformers` 5.0.0 breaks the model  ← the real landmine
Kaggle ships transformers **5.0.0**; the repo pins **4.46.2**.

`bert_lp.py:140` does:
```python
layer_outputs = encoder_layer(hidden_states, attention_mask=...)
hidden_states = layer_outputs[0]
mp_hidden_states = torch.bmm(dense_adj, hidden_states)
```
- transformers **4.x**: `BertLayer` returns a **tuple** → `[0]` is the (B,S,H) tensor. Correct.
- transformers **5.x**: `BertLayer` returns a **tensor** → `[0]` slices a dimension → 2D
  → `RuntimeError: batch2 must be a 3D tensor`.

This hits PENCIL's *multiplicative residual* — the one component the paper claims as novel.

**Fix:** pin `transformers==4.46.2`, plus a version-agnostic guard:
```python
hidden_states = (layer_outputs[0]
                 if isinstance(layer_outputs, (tuple, list)) else layer_outputs)
```

> Note: a plain `import` check does **not** catch this. The module imports fine under 5.0.0
> and only fails at the first forward pass.

### 2. CUDA OOM — an 8×A100 config on one T4
Official config is `batch_size_training: 256` × `gradient_accumulation_steps: 8`
= **effective batch 2048**, tuned for 8×A100 (80 GB each). OOMs at batch 4/36 on a 14.6 GB T4.

**Fix:** `batch_size_training: 64`, `gradient_accumulation_steps: 32`.
Effective batch stays **2048** — identical optimisation, just fits in memory.

### 3. The data archive is 7.5 GB
`data.zip` on Google Drive is **7,533,193,545 bytes**. It filled Kaggle's disk
(`OSError: [Errno 28] No space left on device`). It is *not* quota-blocked — the interstitial
is Google's virus-scan warning for large files.

**Fix:** the server supports HTTP range requests (`Accept-Ranges: bytes`), so only the cora
subset was pulled — **206 MB in 39 seconds** instead of 7.5 GB — and published as the Kaggle
dataset **`ask812/pencil-cora`**. Kernels now attach it; no download at all.

## Cost model (confirmed by the full run)

- **78 s/epoch** measured in the spike; full 150-epoch run took **189.6 min** end to end.
- Comfortably inside Kaggle's 12 h limit — a full cora baseline costs ~3.2 h of free compute.

## Bonus finding

`data/heuristics/cora/aa/all_edge_labels_normalized.npy` — **Adamic–Adar is precomputed** in the
archive, along with `cn`, `ra`, `katz`, `pagerank`, `shortest_path`. The cheap first-stage
retriever for the hybrid does not need to be built from scratch. `data/heart/cora/` also ships
the HeaRT evaluation splits.

## Reproduce

```bash
# spike (~6 min)
cd kernel && python -m kaggle kernels push -p . --accelerator NvidiaTeslaT4

# full baseline (~3.2 h)
cd kernel_baseline && python -m kaggle kernels push -p . --accelerator NvidiaTeslaT4
python wait_for_kernel.py ask812/pencil-baseline 5
python fetch_log.py ask812/pencil-baseline
```
Regenerate kernels with `python _build_kernel.py` / `python _build_baseline.py`.

## Getting the trained checkpoint

`best_model_checkpoint.pth` (316 MB) is in the `ask812/pencil-baseline` kernel output:

```bash
python -m kaggle kernels output ask812/pencil-baseline -p ckpt --file-pattern ".*\.pth"
python verify_checkpoint.py
```
(The 316 MB transfer may drop mid-download — just re-run, it resumes cleanly.)

**Verified locally** — loads, and the weights are real:

| | |
|---|---|
| tensor entries | 147 |
| total params | **28,184,577** (matches the paper's ~27.3M claim) |
| `mp_proj` | 16 tensors — the multiplicative residual, PENCIL's novel component |
| `input_proj` | (1686, 512) orthogonal projection |
| `lp_head` | (1, 1024) |
| NaN / all-zero tensors | **0** |
| also contains | `optimizer_state_dict`, `epoch`, `step`, `best_score` → training is resumable |

> ⚠️ **Gotcha:** every key carries a `module.` prefix because it was saved from a DDP-wrapped
> model. To load it for single-process inference, strip it first:
> ```python
> sd = {k.removeprefix("module."): v for k, v in ck["model_state_dict"].items()}
> ```

**The k-axis of the project needs no retraining** — how many candidates you re-rank is a pure
inference choice. This one checkpoint unlocks most of the planned experiments.


---

# Retrieve-then-rerank budget study (11 Oct 2026, cora)

**Pipeline.** Adamic-Adar (AA, on the train graph, ties broken by degree product) orders every
candidate; PENCIL re-scores only the top k. A PENCIL score does not depend on k, so each sampling
budget s is scored once (`kernel_eval/`, Kaggle `ask812/pencil-budget-eval`, 7.5 min) and every k
is evaluated locally (`analysis.py`). s = (hops, neighbours) is overridden at eval time only;
the input width (max_num_nodes = 842) stays fixed, so the checkpoint loads unchanged.

**Checks passed.**
- full-budget re-eval: test MRR 0.43996 = training's 0.4400
- our AA / CN MRR = 31.87 / 20.99 = the HeaRT benchmark's published cora numbers
- `analysis.py` metric vs repo evaluator: max |diff| 7e-8 over 36 dumps
- invariant asserted: hybrid at k = N equals full PENCIL

| Setting (cora) | test MRR | valid MRR | compute |
|---|---|---|---|
| AA alone (degree tie-break) | 33.2 | 38.2 | 0% |
| AA top-20% -> PENCIL, s = 2x5 | 38.4 +- 0.1 | 43.1 +- 0.1 | 7% |
| All -> PENCIL, s = 2x10 | 42.9 +- 1.9 | 59.7 +- 0.5 | 67% |
| All -> PENCIL, s = 2x5 | 42.9 +- 4.1 | 52.4 +- 2.8 | 36% |
| Full PENCIL, s = 2x20 | 43.7 +- 0.9 | 59.8 +- 0.3 | 100% |

(3 eval seeds; compute = measured T4 forward time per candidate on the test pass.)

**Findings.**
1. Re-ranking the top <= 10% changes nothing: AA's top picks are almost all true links, and
   under this protocol a positive's rank only counts negatives above it.
2. 57% of true test links have zero common neighbours (98.9% of negatives too) - the zone where
   AA is blind and PENCIL earns its cost.
3. k and s interact: 1-hop PENCIL peaks at k = 20% (35.2) then falls to 20.3 at k = N.
4. s = 2x10 matches full PENCIL on both splits; s = 2x5 is cheaper but seed std rises to 4.1.

**Caution.** "7% compute -> 88% of MRR" holds on test (38.4/43.7) but NOT on valid (43.1/59.8 =
72%) - do not quote it. Settings were read off the sweep; final report must select on valid.

**Bug found and fixed on the way.** NumPy 2 keeps `1e9 + float32_array` in float32 (resolution 64
at 1e9), so every re-ranked score tied. Fixed with float64 and an offset of 1 instead of 1e9.

**Reproduce.**
```
python _build_eval.py; cd kernel_eval; python -m kaggle kernels push -p . --accelerator NvidiaTeslaT4
python wait_for_kernel.py ask812/pencil-budget-eval 3
python -m kaggle kernels output ask812/pencil-budget-eval -p evals_dl --file-pattern "evals_pack.zip"
Expand-Archive evals_dl\evals_pack.zip -DestinationPath evals
python analysis.py; python make_figures.py; python _build_slide.py
```

---

# Extension (14-15 Oct 2026): second graph, hard negatives, validation-selected rule

**New runs (all on free Kaggle T4s).**
- `ask812/pencil-citeseer`: official citeseer config (2 layers, 1024 hidden), 2 x T4 DDP, same
  effective batch 2048, 150 epochs in 59 min. Validation-selected **test MRR 50.46** (paper 47.51 +- ~9).
- `ask812/pencil-budget-eval-heart` (cora) and `-citeseer-heart`: the trained checkpoints re-scored
  under HeaRT (each test positive vs its own 500 hard negatives; ~264k / ~228k candidates per pass),
  s in {2x20, 2x10, 2x5, 2x2, 1x20} x 2 seeds. 74 and 52 min.
- `ask812/pencil-budget-eval-citeseer`: pool protocol sweep, same grid as cora.

**Per-query re-ranking under HeaRT.** AA shortlists each query's 501 candidates; PENCIL re-ranks the
top k of that row (k in 0..501). This is the realistic retrieve-then-rerank setting.

**Bug found and fixed: positional tie bias.** The first hybrid code ordered the retriever's ties by
list position, and positives come first in every list, so a tied positive always won. Effect on the
pool numbers was at most 0.2 MRR (AA tie-break 33.24 -> 33.09 test); under HeaRT it was 2 points
(14.15 vs 11.97) because many hard negatives tie with the positive on AA. Now: shortlist ties are
broken at random (seeded) and the un-reranked tail keeps its AA score so its ties stay ties.
`analysis.py` asserts k = 0 equals AA and k = all equals full PENCIL for every dump.

**Checks.** AA test MRR (ties averaged): cora 31.87 pool / 11.91 HeaRT, citeseer 29.37 pool / 10.82
HeaRT - all four match the HeaRT benchmark's published AA numbers. Our metric vs the repo evaluator:
max |diff| 2e-8..1e-7 across all 112 dumps. Full PENCIL under HeaRT on citeseer: 17.8 (paper 16.80).

| (test MRR) | Cora pool | Citeseer pool | Cora HeaRT | Citeseer HeaRT |
|---|---|---|---|---|
| AA alone (degree tie-break) | 33.1 | 31.5 | 12.0 | 11.0 |
| validation-picked, <= 10% compute | 38.4 (7%) | 34.7 (9%) | 12.8 (0.4%) | 13.8 (8%) |
| PENCIL s = 2x10, all candidates | 42.9 (67%) | 51.4 (63%) | 14.3 (65%) | 14.2 (69%) |
| PENCIL s = 2x20 (official) | 43.7 | 49.7 | 13.3 | 17.8 |

Full tables per compute cap (validation pick vs test oracle) and the transfer test:
`python select_rule.py` -> `results/selection.json`.

**What holds and what does not.**
- s = 2x10 is within 1 point of the official s, or better, in 3 of 4 settings; it loses 3.6 on
  citeseer-HeaRT. The best s is task-dependent.
- Validation picks on tiny graphs are noisy: on citeseer-pool at a 10% cap validation picks 34.7 while
  the best test setting gives 45.2. Cora's picks applied to citeseer-pool beat citeseer's own at every
  cap up to 50% (e.g. 44.9 vs 37.9 at 25%). Under HeaRT the transfer holds at small caps only.
- Under HeaRT, MRR is not monotone in k: re-ranking the top 1% of each query recovers most of PENCIL's
  cora gain; re-ranking 2-40% can be worse than 1%.
- Compute here is PENCIL forward time per candidate (batch 64 pool / 256 HeaRT), averaged over seeds;
  AA itself costs ~1 ms per thousand candidates on a laptop CPU and is not counted.
