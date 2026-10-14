"""Retrieve-then-rerank budget study.

Stage 1: Adamic-Adar (AA) on the TRAIN graph orders candidates (ties broken by degree product).
Stage 2: PENCIL re-scores only the top k. A PENCIL score does not depend on k, so each sampling
budget s needs one scoring pass on Kaggle (kernel_eval*); every k is evaluated here for free.

Two protocols:
  pool   planetoid split: every test positive is ranked against ALL test negatives (one shared
         pool, N = P + Q candidates). k is a fraction of the pool.
  heart  HeaRT (Li et al., NeurIPS 2023): every test positive has its OWN 500 hard negatives.
         AA shortlists per query, PENCIL re-ranks the top k of those 501. This is the realistic
         retrieve-then-rerank setting.

Usage: python analysis.py <dataset> <pool|heart>     reads evals/<dataset>_<protocol>/<tag>/*.pt
"""
import csv, glob, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = {"cora": os.path.join(HERE, "dl", "cora_subset", "data"),
         "citeseer": os.path.join(HERE, "dl", "planetoid_subset", "data"),
         "pubmed": os.path.join(HERE, "dl", "planetoid_subset", "data")}
POOL_FRACS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.67, 1.0]
HEART_KS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 501]


def load_split(ds, split):
    d = os.path.join(ROOTS[ds], ds)
    pos = np.loadtxt(os.path.join(d, f"{split}_pos.txt"), dtype=np.int64)
    neg = np.loadtxt(os.path.join(d, f"{split}_neg.txt"), dtype=np.int64)
    return pos, neg


def load_heart(ds, split):
    return np.load(os.path.join(ROOTS[ds], "heart", ds, f"heart_{split}_samples.npy"))


def num_nodes(ds):
    n = max(int(load_heart(ds, s).max()) for s in ("valid", "test"))
    for s in ("train", "valid", "test"):
        n = max([n] + [int(a.max()) for a in load_split(ds, s)])
    return n + 1


def train_graph(ds):
    n = num_nodes(ds)
    tr = load_split(ds, "train")[0]
    tr = tr[tr[:, 0] != tr[:, 1]]
    A = sp.coo_matrix((np.ones(len(tr)), (tr[:, 0], tr[:, 1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.float64).tocsr()


def heuristics(A, edges):
    deg = np.asarray(A.sum(1)).ravel()
    w = np.zeros_like(deg)
    w[deg > 1] = 1.0 / np.log(deg[deg > 1])
    u, v = edges[:, 0], edges[:, 1]
    aa = np.asarray((A @ sp.diags(w))[u].multiply(A[v]).sum(1)).ravel()
    cn = np.asarray(A[u].multiply(A[v]).sum(1)).ravel()
    pa = deg[u] * deg[v]
    return aa, cn, pa


def mrr_hits(score, y):
    """Pool protocol: each positive vs all negatives; ties count half (repo convention)."""
    pos, neg = score[y == 1], score[y == 0]
    opt = (neg[None, :] > pos[:, None]).sum(1)
    pes = (neg[None, :] >= pos[:, None]).sum(1)
    rank = 0.5 * (opt + pes) + 1
    out = {"mrr": float((1.0 / rank).mean())}
    s_neg = np.sort(neg)
    for K in (20, 100):
        out[f"hits@{K}"] = float((pos > s_neg[-K]).mean())
    return out


def mrr_query(pos, neg):
    """HeaRT protocol: pos (Q,), neg (Q, M); each positive vs its own negatives."""
    opt = (neg > pos[:, None]).sum(1)
    pes = (neg >= pos[:, None]).sum(1)
    rank = 0.5 * (opt + pes) + 1
    return {"mrr": float((1.0 / rank).mean()), "hits@1": float((pes == 0).mean()),
            "hits@10": float((pes < 10).mean()), "hits@20": float((pes < 20).mean())}


def hybrid_score(key, pencil, k, rng):
    """key: retriever score (higher = better). Top-k (ties broken at random) get PENCIL scores;
    the tail keeps its retriever key, so ties there stay ties and the metric averages them."""
    n = len(pencil)
    score = key.astype(np.float64) - float(key.max()) - 1.0  # tail scores <= -1
    if k:
        order = np.lexsort((rng.random(n), -key))
        top = order[:k]
        p = pencil[top].astype(np.float64)
        score[top] = 1.0 + (p - p.min())  # >= 1 > tail; a huge offset would round into ties
    return score


def hybrid_query(key, pencil, k, rng):
    """Row-wise hybrid_score. key, pencil: (Q, C). Column 0 is the positive."""
    Q, C = key.shape
    score = key.astype(np.float64) - key.max(1, keepdims=True).astype(np.float64) - 1.0
    if k:
        # random tie-break inside each row: sort by (-key, noise)
        noise = rng.random((Q, C))
        order = np.lexsort((noise.ravel(), -key.ravel(), np.repeat(np.arange(Q), C))).reshape(Q, C)
        top = order[:, :k] - (np.arange(Q) * C)[:, None]
        p = np.take_along_axis(pencil, top, 1).astype(np.float64)
        np.put_along_axis(score, top, 1.0 + (p - p.min(1, keepdims=True)), 1)
    return score


def tie_break(aa, pa):
    """Exact sortable key: AA first, then degree product (dense integer ranks, no float mixing)."""
    ra = np.unique(np.round(aa, 9), return_inverse=True)[1].astype(np.int64)
    rp = np.unique(pa, return_inverse=True)[1].astype(np.int64)
    return ra * (rp.max() + 1) + rp


def load_dumps(ds, protocol):
    runs = {}
    for f in sorted(glob.glob(os.path.join(HERE, "evals", f"{ds}_{protocol}", "*", "*.pt"))):
        tag = os.path.basename(os.path.dirname(f))
        split = os.path.splitext(os.path.basename(f))[0]
        budget, seed = tag.rsplit("_s", 1)
        runs.setdefault((budget, split), []).append((int(seed), torch.load(f, weights_only=False)))
    return runs


def cost_row(d, n):
    return {"fwd_ms_per_cand": d["fwd_s"] * 1e3 / n, "wall_ms_per_cand": d["wall_s"] * 1e3 / n,
            "peak_mb": d["peak_mem"] / 2**20,
            "real_tokens": float(np.mean(d["real_tokens"])) if d["real_tokens"] else None,
            "padded_seq": float(np.mean([s for _, s in d["seq"]])) if d["seq"] else None}


def run_pool(ds, A, runs):
    rows, cost, sanity = [], {}, {}
    for (budget, split), seeds in sorted(runs.items()):
        for seed, d in seeds:
            edges = d["edges"][:, :2].numpy().astype(np.int64)
            y = d["y_true"].numpy().astype(int).ravel()
            pencil = d["y_pred"].numpy().ravel()
            if d["edges"].shape[1] == 3:
                assert (d["edges"][:, 2].numpy() == y).all(), "label/edge misalignment"
            n = len(y)
            full = mrr_hits(pencil, y)
            sanity[f"{budget}_s{seed}_{split}"] = {"mine": full["mrr"],
                                                   "repo": (d["metrics"] or {}).get("mrr")}
            aa, cn, pa = heuristics(A, edges)
            key = tie_break(aa, pa)
            per = d["fwd_s"] * 1e3 / n
            cost.setdefault((budget, split), []).append(cost_row(d, n))
            rng = np.random.default_rng(seed)
            rand_key = rng.permutation(n)
            order = np.lexsort((rng.random(n), -key))
            rank_of = np.empty(n, dtype=int); rank_of[order] = np.arange(n)
            for f in POOL_FRACS:
                k = int(round(f * n))
                m = mrr_hits(hybrid_score(key, pencil, k, np.random.default_rng([seed, k])), y)
                if k == n:
                    assert abs(m["mrr"] - full["mrr"]) < 1e-9, "k=N must equal full PENCIL"
                if k == 0:
                    assert abs(m["mrr"] - mrr_hits(key, y)["mrr"]) < 1e-9, "k=0 must equal AA"
                rows.append({"split": split, "budget": budget, "seed": seed, "frac": f, "k": k,
                             "n": n, **m,
                             "rand_mrr": mrr_hits(hybrid_score(rand_key, pencil, k, rng), y)["mrr"],
                             "aa_recall_at_k": float((rank_of[y == 1] < k).mean()) if k else 0.0,
                             "pencil_ms": k * per})
    base = {}
    for split in ("valid", "test"):
        pos, neg = load_split(ds, split)
        e = np.vstack([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
        aa, cn, pa = heuristics(A, e)
        base[split] = {"AA": mrr_hits(aa, y), "CN": mrr_hits(cn, y), "PA": mrr_hits(pa, y),
                       "AA_tiebreak": mrr_hits(tie_break(aa, pa), y),
                       "chance_mrr": float(np.mean(1.0 / np.arange(1, len(neg) + 2))),
                       "frac_pos_with_zero_cn": float((cn[y == 1] == 0).mean())}
    return rows, cost, sanity, base


def run_heart(ds, A, runs):
    rows, cost, sanity, cache = [], {}, {}, {}
    for (budget, split), seeds in sorted(runs.items()):
        if split not in cache:
            pos = load_split(ds, split)[0]
            neg = load_heart(ds, split)                       # (Q, 500, 2)
            Q, M, _ = neg.shape
            cand = np.concatenate([pos[:, None, :], neg], 1)  # (Q, 501, 2), column 0 = positive
            aa, cn, pa = heuristics(A, cand.reshape(-1, 2))
            cache[split] = (Q, M, tie_break(aa, pa).reshape(Q, M + 1),
                            aa.reshape(Q, M + 1), cn.reshape(Q, M + 1))
        Q, M, retr, aa_q, cn_q = cache[split]
        for seed, d in seeds:
            y = d["y_true"].numpy().astype(int).ravel()
            yp = d["y_pred"].numpy().ravel()
            assert len(yp) == Q * (M + 1) and y[:Q].all() and not y[Q:].any(), "HeaRT layout"
            pencil = np.concatenate([yp[:Q, None], yp[Q:].reshape(Q, M)], 1)
            full = mrr_query(pencil[:, 0], pencil[:, 1:])
            sanity[f"{budget}_s{seed}_{split}"] = {"mine": full["mrr"],
                                                   "repo": (d["metrics"] or {}).get("mrr")}
            per = d["fwd_s"] * 1e3 / len(yp)  # unique-edge scoring; per candidate slot
            cost.setdefault((budget, split), []).append(cost_row(d, len(yp)))
            rng = np.random.default_rng(seed)
            rand = rng.permutation(Q * (M + 1)).reshape(Q, M + 1)
            noise = rng.random(retr.shape)
            pos_rank = ((retr[:, 1:] > retr[:, :1]) |
                        ((retr[:, 1:] == retr[:, :1]) & (noise[:, 1:] > noise[:, :1]))).sum(1)
            for k in HEART_KS:
                s = hybrid_query(retr, pencil, k, np.random.default_rng([seed, k]))
                m = mrr_query(s[:, 0], s[:, 1:])
                if k == M + 1:
                    assert abs(m["mrr"] - full["mrr"]) < 1e-9, "k=all must equal full PENCIL"
                if k == 0:
                    assert abs(m["mrr"] - mrr_query(retr[:, 0], retr[:, 1:])["mrr"]) < 1e-9
                r = hybrid_query(rand, pencil, k, rng)
                rows.append({"split": split, "budget": budget, "seed": seed, "frac": k / (M + 1),
                             "k": k, "n": M + 1, **m,
                             "rand_mrr": mrr_query(r[:, 0], r[:, 1:])["mrr"],
                             "aa_recall_at_k": float((pos_rank < k).mean()) if k else 0.0,
                             "pencil_ms": k * per})
    base = {}
    for split in ("valid", "test"):
        if split not in cache:
            continue
        Q, M, retr, aa_q, cn_q = cache[split]
        base[split] = {"AA": mrr_query(aa_q[:, 0], aa_q[:, 1:]),
                       "CN": mrr_query(cn_q[:, 0], cn_q[:, 1:]),
                       "AA_tiebreak": mrr_query(retr[:, 0], retr[:, 1:]),
                       "chance_mrr": float(np.mean(1.0 / np.arange(1, M + 2))),
                       "frac_pos_with_zero_cn": float((cn_q[:, 0] == 0).mean()),
                       "frac_neg_with_zero_cn": float((cn_q[:, 1:] == 0).mean())}
    return rows, cost, sanity, base


def main(ds, protocol):
    out = os.path.join(HERE, "results", f"{ds}_{protocol}")
    os.makedirs(out, exist_ok=True)
    runs = load_dumps(ds, protocol)
    assert runs, f"no dumps under evals/{ds}_{protocol}"
    A = train_graph(ds)
    t0 = time.time()
    rows, cost, sanity, base = (run_pool if protocol == "pool" else run_heart)(ds, A, runs)
    with open(os.path.join(out, "ksweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = {f"{b}|{s}": {k: float(np.mean([c[k] for c in v if c[k] is not None]))
                        for k in v[0] if any(c[k] is not None for c in v)}
           for (b, s), v in cost.items()}
    json.dump({"dataset": ds, "protocol": protocol, "baselines": base, "cost": agg,
               "sanity": sanity}, open(os.path.join(out, "summary.json"), "w"), indent=1)
    worst = max(abs(v["mine"] - v["repo"]) for v in sanity.values() if v["repo"] is not None)
    print(f"[{ds}/{protocol}] AA test MRR {base['test']['AA']['mrr']*100:.2f} | "
          f"metric vs repo max diff {worst:.1e} over {len(sanity)} dumps | "
          f"{len(rows)} rows in {time.time()-t0:.0f}s -> results/{ds}_{protocol}/")
    assert worst < 1e-4, "our metric disagrees with the repo evaluator"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cora", sys.argv[2] if len(sys.argv) > 2 else "pool")
