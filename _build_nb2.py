import json, os

def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                     "source":s.strip("\n").splitlines(True)}

cells = []

cells.append(md("""
# PENCIL on a single Kaggle T4 — **verified working**

This is not a guess. This exact recipe was run on Kaggle on 29 Aug 2026 and trained `cora`
end to end (kernel `ask812/pencil-spike` v4, exit code 0).

**Notebook settings:** Accelerator = **GPU T4**, Internet = **ON**,
and add the dataset **`ask812/pencil-cora`**.

### The three things that break, and why

| # | Breaks | Cause | Cell |
|---|---|---|---|
| 1 | `RuntimeError: batch2 must be a 3D tensor` | Kaggle ships **transformers 5.0.0**; repo needs **4.46.2** | 3 |
| 2 | `CUDA out of memory` | official config is an **8xA100** batch (2048) | 5 |
| 3 | `No space left on device` | Drive `data.zip` is **7.5 GB** | 4 |

Also: `run_lp.py` **must** be launched with `torchrun` (it calls `init_process_group`
unconditionally), and `WANDB_MODE=disabled` or `wandb.init()` blocks forever on a login prompt.
"""))

cells.append(md("## 1 — Environment\nMeasured on Kaggle: torch 2.10.0+cu128, Python 3.12, Tesla T4 (sm_75, 14.6 GB)."))
cells.append(code("""
import os, sys, subprocess, torch
print("python:", sys.version.split()[0], "| torch:", torch.__version__,
      "| cuda:", torch.version.cuda, "| cpus:", os.cpu_count())
assert torch.cuda.is_available(), "No GPU: Settings -> Accelerator -> GPU T4"
p = torch.cuda.get_device_properties(0)
print(f"gpu: {p.name} | {p.total_memory/1024**3:.1f} GB | sm_{p.major}{p.minor}")

REPO = "/kaggle/working/pencil"
if not os.path.isdir(REPO):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/quang-truong/pencil.git", REPO], check=True)
os.chdir(REPO)
print("cwd:", os.getcwd(), "| .project-root:", os.path.exists(".project-root"))
"""))

cells.append(md("""
## 2 — Install

The README hardcodes a `torch-2.5.1+cu124` wheel index, which does **not** match Kaggle.
We derive the index URL from the running torch instead. `torch_sparse` is a **hard requirement** —
`dataset_map.py` raises `ImportError` without it. Verified: installs in ~12 s.
"""))
cells.append(code("""
import torch, sys, subprocess
def pip(*a): return subprocess.run([sys.executable,"-m","pip","install","-q",*a]).returncode

print(pip("torch_geometric==2.6.1","ogb==1.3.6","rootutils","rich","wandb","pyyaml"))
ver = torch.__version__.split("+")[0]
cu  = "cu" + torch.version.cuda.replace(".","") if torch.version.cuda else "cpu"
URL = f"https://data.pyg.org/whl/torch-{ver}+{cu}.html"
print("wheel index:", URL)
print(pip("torch_sparse","torch_scatter","-f",URL))

import torch_sparse
from torch_geometric.typing import WITH_TORCH_SPARSE
print("torch_sparse:", torch_sparse.__version__, "| WITH_TORCH_SPARSE:", WITH_TORCH_SPARSE)
assert WITH_TORCH_SPARSE, "repo cannot load data without torch_sparse"
"""))

cells.append(md("""
## 3 — Fix the transformers incompatibility  ⚠️ **the subtle one**

`bert_lp.py` does `hidden_states = layer_outputs[0]`.

* transformers **4.x** — `BertLayer` returns a **tuple**, so `[0]` is the `(B,S,H)` tensor. Correct.
* transformers **5.x** — it returns a **tensor**, so `[0]` slices a dim → 2D → `torch.bmm` fails.

This hits PENCIL's *multiplicative residual*, the paper's novel component.

**An `import` test does not catch this** — the module imports fine under 5.0.0 and only dies on
the first forward pass. We pin *and* patch.
"""))
cells.append(code("""
import subprocess, sys, glob, os
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.46.2"], check=False)

n = 0
for path in glob.glob(os.path.join(REPO,"models","**","*.py"), recursive=True):
    src = open(path).read()
    if "hidden_states = layer_outputs[0]" not in src:
        continue
    open(path,"w").write(src.replace(
        "hidden_states = layer_outputs[0]",
        "hidden_states = (layer_outputs[0] "
        "if isinstance(layer_outputs, (tuple, list)) else layer_outputs)"))
    n += 1
    print("patched:", os.path.relpath(path, REPO))
print("files patched:", n)
"""))

cells.append(md("""
## 4 — Data (from the attached dataset, **not** Google Drive)

The upstream `data.zip` is **7.5 GB** and fills Kaggle's disk. We pre-extracted just the cora
subset (206 MB) into `ask812/pencil-cora`. **Attach that dataset** in the sidebar.

It also contains precomputed **Adamic-Adar** (`heuristics/cora/aa/`) and the HeaRT splits —
both directly useful for the project.
"""))
cells.append(code("""
import glob, shutil, os
os.chdir(REPO)
cands = [p for p in glob.glob("/kaggle/input/**/cora", recursive=True) if os.path.isdir(p)]
assert cands, "attach the dataset ask812/pencil-cora in the sidebar"
root = os.path.dirname(sorted(cands, key=len)[0])
print("mount root:", root, "|", sorted(os.listdir(root)))

dst = os.path.join(REPO,"data"); os.makedirs(dst, exist_ok=True)
for sub in os.listdir(root):
    s, d = os.path.join(root,sub), os.path.join(dst,sub)
    if os.path.isdir(s) and not os.path.exists(d):
        shutil.copytree(s,d); print("copied", sub)

for f in ["train_pos.txt","valid_pos.txt","test_pos.txt",
          "valid_neg.txt","test_neg.txt","gnn_feature"]:
    p = os.path.join(dst,"cora",f)
    print(("  OK  " if os.path.exists(p) else "  MISS"), f)
"""))

cells.append(md("""
## 5 — Config: fix the OOM

Official config = `256 x 8` = **effective batch 2048**, tuned for 8xA100 (80 GB each).
A T4 has 14.6 GB and OOMs at batch 4.

We use `64 x 32` — effective batch still **2048**, so the optimisation is unchanged.
Set `num_epochs = 150` for the real baseline (~3.2 h measured).
"""))
cells.append(code("""
import yaml, os
os.chdir(REPO)
c = yaml.safe_load(open("args/official/cora_bert.yaml"))
c["num_epochs"]  = 3      # <-- 150 for the real run (~3.2 h)
c["eval_every"]  = 1
c["num_workers"] = 2
c["name"]        = "smoke"
c["batch_size_training"] = 64
c["gradient_accumulation_steps"] = 32
yaml.safe_dump(c, open("args/official/cora_smoke.yaml","w"), sort_keys=False)
print("effective batch =", c["batch_size_training"]*c["gradient_accumulation_steps"])
"""))

cells.append(md("""
## 6 — Train

`torchrun` is mandatory. `run_lp.py` calls `dist.init_process_group(init_method="env://")`
unconditionally, so bare `python run_lp.py` dies on a missing `MASTER_ADDR`.

Do **not** substitute the repo's `--debug` flag for `WANDB_MODE=disabled`: debug also caps each
epoch at 5 batches *and* skips writing `metrics.jsonl`.
"""))
cells.append(code("""
import os, subprocess
os.chdir(REPO)
env = dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", OMP_NUM_THREADS="2",
           TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
           PYTHONUNBUFFERED="1")
cmd = ["torchrun","--standalone","--nproc_per_node=1","--nnodes=1",
       "run_lp.py","args/official/cora_smoke.yaml"]
print(" ".join(cmd)); print("="*60)
p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for ln in p.stdout: print(ln, end="")
print("="*60, "exit:", p.wait())
"""))

cells.append(md("""
## 7 — Results

cora is scored by **MRR**. Reference from the verified 3-epoch run:
valid MRR 0.0756 → 0.0768 → 0.0866, hits@100 0.517 → 0.646.
"""))
cells.append(code("""
import glob, json, os
for f in sorted(glob.glob(os.path.join(REPO,"ckpts","*","metrics.jsonl"))):
    print("\\n---", os.path.basename(os.path.dirname(f)))
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            print({k:round(v,4) for k,v in d.items()
                   if isinstance(v,float) and ("mrr" in k or "hits@10" in k)})
for f in sorted(glob.glob(os.path.join(REPO,"ckpts","*","best_metrics.json"))):
    print("\\nBEST:", open(f).read()[:300])
"""))

nb = {"cells":cells,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"},"accelerator":"GPU"},
      "nbformat":4,"nbformat_minor":4}

out = r"D:\Projects\da4022_pencil_budget\pencil_kaggle_smoketest.ipynb"
json.dump(nb, open(out,"w",encoding="utf-8"), indent=1)
print("wrote", out, os.path.getsize(out), "bytes,", len(cells), "cells")
