"""Validation-selected budget rule + transfer test.

For each dataset/protocol, a "setting" is (s, k/N). Its compute is k/N x (per-candidate PENCIL
forward time at s) / (same at the official s = 2x20), measured on the test pass.
At each compute cap we pick the setting with the best VALIDATION MRR, then report its TEST MRR
(honest), next to the best test MRR under the cap (oracle, optimistic) and the two anchors.
Transfer: the setting cora's validation picks is applied unchanged to another graph.

Usage: python select_rule.py           -> results/selection.json + printed tables
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CAPS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
FULL = "d2_f20"


def load(tag):
    p = os.path.join(RES, tag)
    if not os.path.exists(os.path.join(p, "ksweep.csv")):
        return None
    R = list(csv.DictReader(open(os.path.join(p, "ksweep.csv"))))
    S = json.load(open(os.path.join(p, "summary.json")))
    cost = {k.split("|")[0]: v["fwd_ms_per_cand"] for k, v in S["cost"].items() if k.endswith("|test")}
    agg = defaultdict(lambda: defaultdict(list))
    for r in R:
        agg[(r["budget"], round(float(r["frac"]), 4))][r["split"]].append(float(r["mrr"]) * 100)
    table = {}
    for (b, f), v in agg.items():
        table[(b, f)] = {"valid": float(np.mean(v["valid"])), "test": float(np.mean(v["test"])),
                         "test_std": float(np.std(v["test"])), "cost": f * cost[b] / cost[FULL]}
    return {"table": table, "summary": S}


def select(table, cap, split="valid"):
    ok = [(k, v) for k, v in table.items() if v["cost"] <= cap + 1e-9]
    return max(ok, key=lambda kv: (kv[1][split], -kv[1]["cost"]))


def fmt(k):
    b, f = k
    s = b.replace("d", "").replace("_f", "x")
    return f"s={s:5s} k={f*100:5.1f}%"


def main():
    tags = [t for t in ("cora_pool", "cora_heart", "citeseer_pool", "citeseer_heart")
            if load(t) is not None]
    out = {}
    for tag in tags:
        D = load(tag); T = D["table"]
        aa = T[(FULL, 0.0)]
        full = T[(FULL, 1.0)]
        print(f"\n== {tag}: AA {aa['test']:.1f} | full PENCIL {full['test']:.1f} +- {full['test_std']:.1f}"
              f" (chance {D['summary']['baselines']['test']['chance_mrr']*100:.1f})")
        print(f"{'cap':>5} | {'validation pick':22s} {'cost':>6} {'test':>12} | {'oracle (test pick)':22s} {'test':>6}")
        rows = []
        for cap in CAPS:
            kv, v = select(T, cap)
            ko, vo = select(T, cap, "test")
            rows.append({"cap": cap, "pick": list(kv), "cost": v["cost"], "valid": v["valid"],
                         "test": v["test"], "test_std": v["test_std"], "oracle_pick": list(ko),
                         "oracle_test": vo["test"]})
            print(f"{cap*100:4.0f}% | {fmt(kv):22s} {v['cost']*100:5.1f}% {v['test']:5.1f} +- {v['test_std']:3.1f}"
                  f" | {fmt(ko):22s} {vo['test']:5.1f}")
        out[tag] = {"aa_test": aa["test"], "full_test": full["test"], "full_std": full["test_std"],
                    "rows": rows}

    for proto in ("pool", "heart"):
        src, dst = f"cora_{proto}", f"citeseer_{proto}"
        if src in out and dst in out:
            Td = load(dst)["table"]
            print(f"\n== transfer {src} -> {dst}: cora's validation pick, applied unchanged")
            tr = []
            for r in out[src]["rows"]:
                k = tuple([r["pick"][0], round(r["pick"][1], 4)])
                own = next(x for x in out[dst]["rows"] if x["cap"] == r["cap"])
                v = Td[k]
                tr.append({"cap": r["cap"], "pick": list(k), "cost_on_target": v["cost"],
                           "test": v["test"], "own_pick_test": own["test"]})
                print(f"{r['cap']*100:4.0f}% | {fmt(k):22s} cost here {v['cost']*100:5.1f}% "
                      f"test {v['test']:5.1f} | citeseer's own pick {own['test']:5.1f}")
            out[f"transfer_{proto}"] = tr
    json.dump(out, open(os.path.join(RES, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
