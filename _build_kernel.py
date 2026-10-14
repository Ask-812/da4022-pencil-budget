import json, os

def code(src):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "source":src.strip("\n").splitlines(True)}

cells = []

# ---- stage harness -------------------------------------------------
cells.append(code(r'''
# ============ STAGE HARNESS ============
# Every stage is wrapped so a failure NEVER aborts the run.
# One kernel run must yield maximum diagnostic information.
import os, sys, subprocess, traceback, time, json
STATE = {}
T0 = time.time()

def stage(name):
    def deco(fn):
        print("\n" + "="*70)
        print(f"### STAGE: {name}   (t+{time.time()-T0:.0f}s)")
        print("="*70, flush=True)
        try:
            r = fn()
            STATE[name] = ("OK", r)
            print(f"--- STAGE {name}: OK", flush=True)
            return r
        except Exception as e:
            STATE[name] = ("FAIL", f"{type(e).__name__}: {e}")
            print(f"--- STAGE {name}: FAIL -> {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None
    return deco

def sh(cmd, env=None, timeout=None):
    """Run a shell command, stream output, return (rc, tail)."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env or os.environ.copy())
    lines = []
    try:
        for ln in p.stdout:
            lines.append(ln)
            print(ln, end="", flush=True)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); rc = -9
        print("!! TIMEOUT", flush=True)
    return rc, "".join(lines[-40:])
'''))

# ---- stage 1: env --------------------------------------------------
cells.append(code(r'''
@stage("01_env")
def _():
    import torch
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
        "cpus": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory/1024**3,1),
                    sm=f"sm_{p.major}{p.minor}", n_gpu=torch.cuda.device_count())
    for k,v in info.items(): print(f"  {k:15s}: {v}")
    # sm_75 = T4 (good). sm_60 = P100 -> known-bad per prior experience.
    return info
'''))

# ---- stage 2: internet ---------------------------------------------
cells.append(code(r'''
@stage("02_internet")
def _():
    import urllib.request
    urllib.request.urlopen("https://pypi.org", timeout=20)
    print("  internet OK")
    return True
'''))

# ---- stage 3: clone ------------------------------------------------
cells.append(code(r'''
REPO = "/tmp/pencil"   # NOT /kaggle/working: keeps kernel output tiny so log fetch is fast

@stage("03_clone")
def _():
    if not os.path.isdir(REPO):
        rc,_ = sh(f"git clone --depth 1 https://github.com/quang-truong/pencil.git {REPO}")
        assert rc == 0, f"clone failed rc={rc}"
    os.chdir(REPO)
    print("  cwd:", os.getcwd())
    print("  .project-root:", os.path.exists(".project-root"))
    print("  configs:", sorted(os.listdir("args/official"))[:4])
    return True
'''))

# ---- stage 4: install ----------------------------------------------
cells.append(code(r'''
@stage("04_install")
def _():
    import torch
    ver = torch.__version__.split("+")[0]
    cu  = "cu" + torch.version.cuda.replace(".","") if torch.version.cuda else "cpu"
    URL = f"https://data.pyg.org/whl/torch-{ver}+{cu}.html"
    print("  wheel index:", URL)

    # pure python first
    rc1,_ = sh(f"{sys.executable} -m pip install -q "
               f"torch_geometric==2.6.1 ogb==1.3.6 rootutils rich wandb gdown pyyaml")
    print("  pure-python rc:", rc1)

    # compiled: try exact index, then fall back to nearest patch versions
    rc2, tail = sh(f"{sys.executable} -m pip install -q torch_sparse torch_scatter -f {URL}")
    print("  compiled rc:", rc2)
    if rc2 != 0:
        base = ver.rsplit(".",1)[0]
        for alt in [f"{base}.0", f"{base}.1"]:
            if alt == ver: continue
            U2 = f"https://data.pyg.org/whl/torch-{alt}+{cu}.html"
            print("  RETRY with", U2)
            rc2, tail = sh(f"{sys.executable} -m pip install -q torch_sparse torch_scatter -f {U2}")
            if rc2 == 0: break
    return {"wheel_url": URL, "rc_pure": rc1, "rc_compiled": rc2}
'''))

# ---- stage 5: verify torch_sparse ----------------------------------
cells.append(code(r'''
@stage("05_verify_torch_sparse")
def _():
    # THE critical dependency: dataset_map.py raises ImportError without it.
    import importlib
    for m in ("torch_sparse","torch_scatter","torch_geometric"):
        importlib.invalidate_caches()
        mod = importlib.import_module(m)
        print(f"  {m:18s}", getattr(mod,"__version__","?"))
    from torch_geometric.typing import WITH_TORCH_SPARSE
    print("  WITH_TORCH_SPARSE:", WITH_TORCH_SPARSE)
    assert WITH_TORCH_SPARSE, "WITH_TORCH_SPARSE is False -> repo cannot load data"
    return True
'''))

# ---- stage 6: transformers version ---------------------------------
cells.append(code(r'''
@stage("06_transformers")
def _():
    # THE BUG (kernel v2): bert_lp.py does
    #     layer_outputs = encoder_layer(...); hidden_states = layer_outputs[0]
    # transformers 4.x BertLayer returns a TUPLE  -> [0] is the (B,S,H) tensor.
    # transformers 5.x BertLayer returns a TENSOR -> [0] slices a dim -> (S,H) 2D
    #                                                -> torch.bmm "batch2 must be 3D".
    # Fix 1: pin to the authors' version (faithful reproduction).
    import transformers
    print("  transformers before:", transformers.__version__)
    rc,_ = sh(f"{sys.executable} -m pip install -q transformers==4.46.2")
    print("  pin rc:", rc)
    return {"pin_rc": rc}
'''))

cells.append(code(r'''
@stage("06b_compat_patch")
def _():
    # Fix 2 (belt and braces): make the indexing version-agnostic.
    # No-op when layer_outputs is already a tuple, so it is safe under 4.46.2 too.
    import re, glob
    n_patched = 0
    for path in glob.glob(os.path.join(REPO, "models", "**", "*.py"), recursive=True):
        src = open(path).read()
        if "layer_outputs[0]" not in src:
            continue
        new = src.replace(
            "hidden_states = layer_outputs[0]",
            "hidden_states = (layer_outputs[0] "
            "if isinstance(layer_outputs, (tuple, list)) else layer_outputs)")
        if new != src:
            open(path, "w").write(new)
            n_patched += 1
            print("  patched:", os.path.relpath(path, REPO))
    print("  files patched:", n_patched)
    return {"patched": n_patched}
'''))

cells.append(code(r'''
@stage("06c_verify_import")
def _():
    # fresh subprocess: the pinned transformers must actually import
    rc, tail = sh(f'cd {REPO} && {sys.executable} -c '
                  f'"import transformers; print(transformers.__version__); '
                  f'import models.transformers.bert_lp as m; print(\'bert_lp OK\')"')
    assert rc == 0, "repo model module does not import under pinned transformers"
    return True
'''))

# ---- stage 7: data (from attached Kaggle dataset, NOT Drive) --------
cells.append(code(r'''
@stage("07_data")
def _():
    # The upstream data.zip on Google Drive is 7.5 GB and blew Kaggle's disk.
    # We pre-extracted ONLY the cora subset (206 MB) into a Kaggle dataset.
    import glob, shutil
    os.chdir(REPO)

    # API-pushed kernels mount inputs at a nested path; search robustly.
    cands = [p for p in glob.glob("/kaggle/input/**/cora", recursive=True)
             if os.path.isdir(p)]
    print("  cora dir candidates:", cands[:10])
    assert cands, "attached dataset not found under /kaggle/input"

    # mount root = the parent that holds cora/ (and maybe heart/, heuristics/)
    root = os.path.dirname(sorted(cands, key=len)[0])
    print("  using mount root:", root)
    print("  contents:", sorted(os.listdir(root))[:10])

    dst = os.path.join(REPO, "data")
    os.makedirs(dst, exist_ok=True)
    for sub in os.listdir(root):
        s = os.path.join(root, sub)
        d = os.path.join(dst, sub)
        if os.path.isdir(s) and not os.path.exists(d):
            shutil.copytree(s, d)
            print(f"  copied {sub}")
    print("  data/ now:", sorted(os.listdir(dst)))
    return True
'''))

cells.append(code(r'''
@stage("08_verify_data")
def _():
    need = ["train_pos.txt","valid_pos.txt","test_pos.txt",
            "valid_neg.txt","test_neg.txt","gnn_feature"]
    d = os.path.join(REPO,"data","cora")
    assert os.path.isdir(d), f"missing {d}"
    missing = []
    for f in need:
        p = os.path.join(d,f)
        ok = os.path.exists(p)
        print(("  OK   " if ok else "  MISS"), f,
              f"({os.path.getsize(p)/1024:.0f} KB)" if ok else "")
        if not ok: missing.append(f)
    assert not missing, f"missing files: {missing}"
    return True
'''))

# ---- stage 9: tiny config ------------------------------------------
cells.append(code(r'''
@stage("09_config")
def _():
    import yaml
    os.chdir(REPO)
    c = yaml.safe_load(open("args/official/cora_bert.yaml"))
    c["num_epochs"]  = 3     # bounded runtime; enough to see MRR move
    c["eval_every"]  = 1     # must be <= num_epochs or we get no metric
    c["num_workers"] = 2     # Kaggle has ~4 cores, config asks 8
    c["name"]        = "smoke"
    # OOM fix: official config is 256 x 8 = 2048 effective batch on 8xA100(80GB).
    # One T4 has 14.6GB. Cut per-device batch 4x and raise accumulation 4x so the
    # EFFECTIVE batch stays 2048 -- i.e. same optimisation, just fits in memory.
    c["batch_size_training"] = 64
    c["gradient_accumulation_steps"] = 32
    yaml.safe_dump(c, open("args/official/cora_smoke.yaml","w"), sort_keys=False)
    print(open("args/official/cora_smoke.yaml").read())
    eff = c["batch_size_training"] * c["gradient_accumulation_steps"]
    print(f"  effective batch = {eff} (official = 256 x 8 = 2048)")
    return True
'''))

# ---- stage 10: TRAIN -----------------------------------------------
cells.append(code(r'''
@stage("10_train")
def _():
    os.chdir(REPO)
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled",       # else wandb.init() blocks on login
               WANDB_SILENT="true",
               OMP_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false",
               PYTORCH_ALLOC_CONF="expandable_segments:True",  # reduce fragmentation
               PYTHONUNBUFFERED="1")
    # MUST use torchrun: run_lp.py calls init_process_group(init_method="env://")
    # unconditionally, so bare `python run_lp.py` dies on missing MASTER_ADDR.
    cmd = ("torchrun --standalone --nproc_per_node=1 --nnodes=1 "
           "run_lp.py args/official/cora_smoke.yaml")
    rc, tail = sh(cmd, env=env, timeout=5400)
    print("\n  exit code:", rc)
    return {"rc": rc, "tail": tail}
'''))

# ---- stage 11: metrics ---------------------------------------------
cells.append(code(r'''
@stage("11_metrics")
def _():
    import glob
    found = {}
    for f in glob.glob(os.path.join(REPO,"ckpts","*","metrics.jsonl")):
        run = os.path.basename(os.path.dirname(f))
        rows = [json.loads(l) for l in open(f) if l.strip()]
        found[run] = rows
        print(f"  {run}: {len(rows)} rows")
        for r in rows[:4]:
            print("   ", {k:v for k,v in r.items()
                          if "mrr" in k.lower() or "hits@10" in k.lower() or "epoch" in k.lower()})
    for f in glob.glob(os.path.join(REPO,"ckpts","*","best_metrics.json")):
        print("  BEST:", open(f).read()[:500])
        found.setdefault("best", []).append(1)
    if not found:
        print("  no metrics found; ckpts tree:")
        for p in glob.glob(os.path.join(REPO,"ckpts","**"), recursive=True)[:40]:
            print("   ", p)
    return found
'''))

# ---- stage 12: verdict ---------------------------------------------
cells.append(code(r'''
print("\n\n" + "#"*70)
print("# SPIKE SUMMARY")
print("#"*70)
for k,(status,val) in STATE.items():
    s = str(val)
    if len(s) > 200: s = s[:200] + "..."
    print(f"  {k:24s} {status:5s}  {s}")

trained = STATE.get("10_train",("FAIL",{}))[1] or {}
metrics = STATE.get("11_metrics",("FAIL",None))[1]
passed  = (STATE.get("05_verify_torch_sparse",("FAIL",))[0]=="OK"
           and isinstance(trained,dict) and trained.get("rc")==0
           and bool(metrics))

print("\n" + "#"*70)
print("# VERDICT:", "PASS -> PENCIL is reproducible on one T4"
                    if passed else "FAIL -> see first FAIL stage above")
print("#"*70)
print(f"total runtime: {time.time()-T0:.0f}s")
'''))

nb = {"cells":cells,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"},"accelerator":"GPU"},
      "nbformat":4,"nbformat_minor":4}

outdir = r"D:\Projects\da4022_pencil_budget\kernel"
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "spike_kernel.ipynb")
json.dump(nb, open(out,"w",encoding="utf-8"), indent=1)

meta = {
    "id": "ask812/pencil-spike",
    "title": "pencil-spike",
    "code_file": "spike_kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["ask812/pencil-cora"],
    "competition_sources": [],
    "kernel_sources": [],
}
json.dump(meta, open(os.path.join(outdir,"kernel-metadata.json"),"w"), indent=2)
print("wrote", out, os.path.getsize(out), "bytes,", len(cells), "cells")
