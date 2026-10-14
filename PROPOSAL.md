# DA4022 — Course Project Proposal

**How Much PENCIL Can You Afford? Compute-Allocation Laws for Link Prediction**

**Team:** Amit Kumar · Ashmit Satpute · Arnav Kshirsagar · Pranav Shalya · Atharv Wankhede
**Type:** Self-proposed, extending our paper presentation — *Plain Transformers are Surprisingly
Powerful Link Predictors* (PENCIL, ICML 2026). Closest anchor: #2.

PENCIL matches heuristic-informed GNNs at link prediction using a plain Transformer over sampled
subgraphs. Its accuracy, though, is bought with compute: Section 7 states that "its GPU compute
and memory scale linearly with the number of candidate links," and the authors name
resource-efficient variants as future work. Expressive power likewise rises with the sampling
budget, so quality is capped by what one can afford *per edge*. Neither the cost of a prediction,
nor the accuracy that cost buys, is ever reported as a first-class result.

We propose to supply that analysis. Production retrieval systems never score every candidate with
an expensive model: a cheap stage shortlists candidates and the expensive model re-ranks the top
*k*. PENCIL's authors suggest exactly this hybrid, citing ContextGNN, and never test it. We will
build it — Adamic–Adar as the first stage, PENCIL as the re-ranker — then treat inference as a
budget-allocation problem between *k*, the number of candidates re-ranked, and *s*, the sampling
budget each one receives. Under a fixed budget, is it better to re-rank many candidates shallowly
or few deeply? We will sweep both on cora, citeseer and ogbl-collab, measuring accuracy against
latency and peak memory, map the accuracy-versus-cost Pareto frontier, and fit a rule for
choosing (*k*, *s*) which we then test on a graph it was not fitted on.

Feasibility is already established: we have reproduced the paper's cora result — test MRR
**44.00** against the published **42.23** — in 3.2 hours on a single free Kaggle T4, where the
authors used four to eight A100-class GPUs. We hold the trained checkpoint, and the *k* axis
needs no retraining, since the number of candidates re-ranked is purely an inference choice.
