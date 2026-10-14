"""Full PENCIL training kernel (proven v4 spike recipe), generic over dataset and GPU count.
python _build_baseline.py                                  -> cora, 1 GPU (the original baseline)
python _build_baseline.py citeseer ask812/pencil-citeseer ask812/pencil-planetoid 2"""
import json, os, sys

DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
SLUG    = sys.argv[2] if len(sys.argv) > 2 else "ask812/pencil-baseline"
DSRC    = sys.argv[3] if len(sys.argv) > 3 else "ask812/pencil-cora"
NPROC   = int(sys.argv[4]) if len(sys.argv) > 4 else 1

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                     "source":s.strip("\n").splitlines(True)}

cells = []

cells.append(code((r'''
# ===== harness (same as the proven spike) =====
import os, sys, subprocess, traceback, time, json
STATE, T0 = {}, time.time()

def stage(name):
    def deco(fn):
        print("\n"+"="*70); print(f"### STAGE: {name}  (t+{time.time()-T0:.0f}s)"); print("="*70, flush=True)
        try:
            r = fn(); STATE[name]=("OK",r); print(f"--- STAGE {name}: OK", flush=True); return r
        except Exception as e:
            STATE[name]=("FAIL",f"{type(e).__name__}: {e}")
            print(f"--- STAGE {name}: FAIL -> {e}", flush=True); traceback.print_exc(); return None
    return deco

def sh(cmd, env=None, timeout=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env or os.environ.copy())
    lines=[]
    try:
        for ln in p.stdout:
            lines.append(ln)
            # throttle tqdm spam: only keep meaningful lines
            if ("metrics" in ln or "Epoch" in ln and "batch 0/" in ln
                    or "Best" in ln or "Error" in ln or "error" in ln or "Traceback" in ln):
                print(ln, end="", flush=True)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); rc=-9; print("!! TIMEOUT", flush=True)
    return rc, "".join(lines[-60:])

REPO = "/tmp/pencil"
DATASET = "__DATASET__"
NPROC = __NPROC__
''').replace("__DATASET__", DATASET).replace("__NPROC__", str(NPROC))))

cells.append(code(r'''
@stage("01_setup")
def _():
    import torch
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name} {p.total_memory/1024**3:.1f}GB sm_{p.major}{p.minor} | torch {torch.__version__}")
    if not os.path.isdir(REPO):
        sh(f"git clone --depth 1 https://github.com/quang-truong/pencil.git {REPO}")
    os.chdir(REPO)

    ver = torch.__version__.split("+")[0]
    cu  = "cu"+torch.version.cuda.replace(".","")
    sh(f"{sys.executable} -m pip install -q torch_geometric==2.6.1 ogb==1.3.6 rootutils rich wandb pyyaml")
    sh(f"{sys.executable} -m pip install -q torch_sparse torch_scatter "
       f"-f https://data.pyg.org/whl/torch-{ver}+{cu}.html")
    sh(f"{sys.executable} -m pip install -q transformers==4.46.2")

    from torch_geometric.typing import WITH_TORCH_SPARSE
    assert WITH_TORCH_SPARSE
    print("  WITH_TORCH_SPARSE OK")
    return True
'''))

cells.append(code(r'''
@stage("02_patch")
def _():
    # transformers 5.x returns a tensor not a tuple; guard is a no-op under 4.46.2
    import glob
    n=0
    for path in glob.glob(os.path.join(REPO,"models","**","*.py"), recursive=True):
        src=open(path).read()
        if "hidden_states = layer_outputs[0]" in src:
            open(path,"w").write(src.replace(
                "hidden_states = layer_outputs[0]",
                "hidden_states = (layer_outputs[0] "
                "if isinstance(layer_outputs, (tuple, list)) else layer_outputs)"))
            n+=1
    print("  patched files:", n)
    return n
'''))

cells.append(code(r'''
@stage("03_data")
def _():
    import glob, shutil
    os.chdir(REPO)
    cands=[p for p in glob.glob(f"/kaggle/input/**/{DATASET}", recursive=True)
           if os.path.exists(os.path.join(p, "gnn_feature"))]
    assert cands, f"{DATASET} data not attached"
    root=os.path.dirname(sorted(cands,key=len)[0])
    dst=os.path.join(REPO,"data"); os.makedirs(dst,exist_ok=True)
    for sub in os.listdir(root):
        s,d=os.path.join(root,sub),os.path.join(dst,sub)
        if os.path.isdir(s) and not os.path.exists(d): shutil.copytree(s,d)
    print("  data/:", sorted(os.listdir(dst)))
    assert os.path.exists(os.path.join(dst,DATASET,"gnn_feature"))
    return True
'''))

cells.append(code(r'''
@stage("04_config")
def _():
    import yaml
    os.chdir(REPO)
    c=yaml.safe_load(open(f"args/official/{DATASET}_bert.yaml"))
    eff=c["batch_size_training"]*c["gradient_accumulation_steps"]
    # FAITHFUL baseline: keep official num_epochs / eval_every / lr / sampling.
    # Only change: per-device batch split (same effective batch) + workers + checkpoint policy.
    c["batch_size_training"]=64          # official 256 OOMs on a 14.6GB T4
    c["gradient_accumulation_steps"]=eff//64  # total steps (cora: 32); run_lp.py splits it across GPUs
    assert (eff//64) % NPROC == 0
    c["num_workers"]=2                   # Kaggle has 4 cores
    c["save_only_improve"]=True          # official False would write 150 checkpoints -> disk blowup
    c["name"]="baseline"
    yaml.safe_dump(c, open(f"args/official/{DATASET}_full.yaml","w"), sort_keys=False)
    print(f"  epochs={c['num_epochs']} eval_every={c['eval_every']} "
          f"eff_batch={c['batch_size_training']*c['gradient_accumulation_steps']}")
    return True
'''))

cells.append(code(r'''
@stage("05_train_full")
def _():
    os.chdir(REPO)
    env=dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", OMP_NUM_THREADS="2",
             TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
             PYTHONUNBUFFERED="1")
    # ~78 s/epoch measured -> 150 epochs ~= 3.2 h. Cap well under Kaggle's 12 h.
    rc,tail = sh(f"torchrun --standalone --nproc_per_node={NPROC} --nnodes=1 "
                 f"run_lp.py args/official/{DATASET}_full.yaml", env=env, timeout=39600)
    if rc != 0: print(tail)
    print("  exit code:", rc)
    return {"rc":rc}
'''))

cells.append(code(r'''
@stage("06_export")
def _():
    # copy ONLY the small artefacts to /kaggle/working so kernel output stays manageable
    import glob, shutil
    out="/kaggle/working/baseline"; os.makedirs(out, exist_ok=True)
    saved=[]
    for d in glob.glob(os.path.join(REPO,"ckpts","*")):
        run=os.path.basename(d); tgt=os.path.join(out,run); os.makedirs(tgt,exist_ok=True)
        for fn in ("metrics.jsonl","best_metrics.json","config.yaml","best_model_checkpoint.pth"):
            src=os.path.join(d,fn)
            if os.path.exists(src):
                shutil.copy2(src,tgt)
                saved.append((run+"/"+fn, os.path.getsize(src)/1024**2))
    for n,s in saved: print(f"  saved {n}  ({s:.1f} MB)")
    return saved
'''))

cells.append(code(r'''
@stage("07_results")
def _():
    import glob
    rows=[]
    for f in glob.glob(os.path.join(REPO,"ckpts","*","metrics.jsonl")):
        for line in open(f):
            if line.strip(): rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("valid/epoch",0))
    print(f"  {len(rows)} eval points")
    for r in rows[::max(1,len(rows)//15)]:
        print(f"   ep{r.get('valid/epoch'):>4}  valid_mrr={r.get('valid/mrr',0):.4f}  "
              f"test_mrr={r.get('test/mrr',0):.4f}  valid_h@100={r.get('valid/hits@100',0):.4f}")
    for f in glob.glob(os.path.join(REPO,"ckpts","*","best_metrics.json")):
        print("\n  BEST:", open(f).read())
    return len(rows)
'''))

cells.append(code(r'''
print("\n"+"#"*70); print("# BASELINE SUMMARY"); print("#"*70)
for k,(s,v) in STATE.items():
    t=str(v); print(f"  {k:18s} {s:5s}  {t[:150]}")
print(f"\ntotal runtime: {(time.time()-T0)/60:.1f} min")
'''))

nb={"cells":cells,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"},"accelerator":"GPU"},
    "nbformat":4,"nbformat_minor":4}

outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "kernel_baseline" if DATASET=="cora" else f"kernel_train_{DATASET}")
os.makedirs(outdir,exist_ok=True)
json.dump(nb, open(os.path.join(outdir,"baseline_kernel.ipynb"),"w",encoding="utf-8"), indent=1)
json.dump({"id":SLUG,"title":SLUG.split("/")[1],
           "code_file":"baseline_kernel.ipynb","language":"python","kernel_type":"notebook",
           "is_private":True,"enable_gpu":True,"enable_internet":True,
           "dataset_sources":[DSRC],
           "competition_sources":[],"kernel_sources":[]},
          open(os.path.join(outdir,"kernel-metadata.json"),"w"), indent=2)
print("wrote", outdir, len(cells), "cells")
