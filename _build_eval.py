"""Eval-only kernel: score every valid/test candidate with a trained checkpoint,
once per sampling budget s, and dump scores + cost. The k-sweep is done locally.

python _build_eval.py                       -> cora, planetoid negatives (the mid-report run)
python _build_eval.py cora 1                -> cora, HeaRT per-query hard negatives
python _build_eval.py citeseer 0|1          -> citeseer (checkpoint from ask812/pencil-citeseer)"""
import json, os, sys

DATASET = sys.argv[1] if len(sys.argv) > 1 else "cora"
HEART = len(sys.argv) > 2 and sys.argv[2] == "1"
CKPT_KERNEL = {"cora": "ask812/pencil-baseline"}.get(DATASET, f"ask812/pencil-{DATASET}")
DATA_SRC = "ask812/pencil-cora" if DATASET == "cora" else "ask812/pencil-planetoid"
SLUG = "ask812/pencil-budget-eval" + ("" if DATASET == "cora" else f"-{DATASET}") + ("-heart" if HEART else "")
if HEART:  # ~200x more candidates per pass: fewer settings, bigger batches
    GRID, SEEDS, BS = "[[[2,20]], [[2,10]], [[2,5]], [[2,2]], [[1,20]]]", "[0, 1]", 256
else:
    GRID, SEEDS, BS = "[[[2,20]], [[2,10]], [[2,5]], [[2,2]], [[1,20]], [[3,5]]]", "[0, 1, 2]", 64

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                     "source":s.strip("\n").splitlines(True)}

cells = []

cells.append(code((r'''
import os, sys, subprocess, traceback, time, json, glob, shutil
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

def sh(cmd, env=None, timeout=None, keep=("DUMPED","metrics","Error","error","Traceback","EVAL override")):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env or os.environ.copy())
    lines=[]
    try:
        for ln in p.stdout:
            lines.append(ln)
            if any(k in ln for k in keep): print(ln, end="", flush=True)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); rc=-9; print("!! TIMEOUT", flush=True)
    return rc, "".join(lines[-40:])

REPO = "/tmp/pencil"
OUT = "/kaggle/working/evals"
DATASET, HEART, BS = "__DATASET__", __HEART__, __BS__
''').replace("__DATASET__", DATASET).replace("__HEART__", str(HEART)).replace("__BS__", str(BS))))

cells.append(code(r'''
@stage("01_setup")
def _():
    import torch
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name} {p.total_memory/1024**3:.1f}GB sm_{p.major}{p.minor} | torch {torch.__version__}")
    if not os.path.isdir(REPO):
        sh(f"git clone --depth 1 https://github.com/quang-truong/pencil.git {REPO}")
    ver = torch.__version__.split("+")[0]
    cu  = "cu"+torch.version.cuda.replace(".","")
    sh(f"{sys.executable} -m pip install -q torch_geometric==2.6.1 ogb==1.3.6 rootutils rich wandb pyyaml")
    sh(f"{sys.executable} -m pip install -q torch_sparse torch_scatter "
       f"-f https://data.pyg.org/whl/torch-{ver}+{cu}.html")
    sh(f"{sys.executable} -m pip install -q transformers==4.46.2")
    from torch_geometric.typing import WITH_TORCH_SPARSE
    assert WITH_TORCH_SPARSE
    return True
'''))

cells.append(code(r'''
WRAPPER = r"""
def evaluate_loop(model, loader, *args, **kwargs):
    # budget-study wrapper: times the eval pass and dumps per-candidate scores
    import time as _time
    split = kwargs.get("split_name", "split")
    seq, real, fwd_t, t_start = [], [], [0.0], [0.0]
    def _pre(mod, a, kw):
        torch.cuda.synchronize(); t_start[0] = _time.perf_counter()
        for v in list(a) + list(kw.values()):
            if torch.is_tensor(v) and v.dim() == 3:
                seq.append((int(v.shape[0]), int(v.shape[1]))); break
        for k, v in kw.items():
            if "attention_mask" in k and torch.is_tensor(v) and v.dim() == 2:
                real.append(float(v.float().sum(dim=1).mean())); break
    def _post(mod, a, kw, out):
        torch.cuda.synchronize(); fwd_t[0] += _time.perf_counter() - t_start[0]
    h1 = model.register_forward_pre_hook(_pre, with_kwargs=True)
    h2 = model.register_forward_hook(_post, with_kwargs=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = _time.perf_counter()
    try:
        out = _evaluate_loop_orig(model, loader, *args, **kwargs)
    finally:
        h1.remove(); h2.remove()
    torch.cuda.synchronize()
    wall = _time.perf_counter() - t0
    dump = os.environ.get("DUMP_DIR")
    if dump:
        os.makedirs(dump, exist_ok=True)
        ds, edges = loader.dataset, None
        for _ in range(6):
            if ds is None:
                break
            if hasattr(ds, "all_edges_with_y"):
                edges = ds.all_edges_with_y.clone(); break
            nxt = getattr(ds, "original_dataset", None)
            ds = nxt if nxt is not None else getattr(ds, "dataset", None)
        yp, yt = out.get("y_pred_all"), out.get("y_true_all")
        torch.save({"y_pred": None if yp is None else yp.detach().float().cpu(),
                    "y_true": None if yt is None else yt.detach().float().cpu(),
                    "edges": edges, "wall_s": wall, "fwd_s": fwd_t[0],
                    "peak_mem": torch.cuda.max_memory_allocated(),
                    "seq": seq, "real_tokens": real, "metrics": out.get("metrics"),
                    "depth_neighbors": os.environ.get("EVAL_DEPTH_NEIGHBORS")},
                   os.path.join(dump, f"{split}.pt"))
        n = 0 if yp is None else len(yp)
        ms = sum(s for _, s in seq) / max(1, len(seq))
        mr = sum(real) / max(1, len(real))
        print(f"DUMPED {split}: wall={wall:.2f}s fwd={fwd_t[0]:.2f}s n={n} padded_seq={ms:.1f} "
              f"real_tokens={mr:.1f} peak={torch.cuda.max_memory_allocated()/2**20:.0f}MB "
              f"edges={'yes' if edges is not None else 'NO'}")
    return out


"""

OVERRIDE = """    if os.environ.get("EVAL_DEPTH_NEIGHBORS"):
        _dn = json.loads(os.environ["EVAL_DEPTH_NEIGHBORS"])
        configs.sampling_config.edge_ego.depth_neighbors = _dn
        print(f"EVAL override depth_neighbors -> {_dn} (max_num_nodes kept at {max_num_nodes})")

"""

@stage("02_patch")
def _():
    # 1) transformers 5.x guard (no-op under 4.46.2)
    n=0
    for path in glob.glob(os.path.join(REPO,"models","**","*.py"), recursive=True):
        src=open(path).read()
        if "hidden_states = layer_outputs[0]" in src:
            open(path,"w").write(src.replace("hidden_states = layer_outputs[0]",
                "hidden_states = (layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs)"))
            n+=1
    # 2) run_lp.py: wrap evaluate_loop + allow sampling-budget override at eval time
    f=os.path.join(REPO,"run_lp.py"); src=open(f).read()
    assert src.count("def evaluate_loop(")==1 and src.count("def train_loop(")==1
    anchor="    # Override only_eval if only_eval argument is provided"
    assert src.count(anchor)==1
    src=src.replace("def evaluate_loop(","def _evaluate_loop_orig(")
    src=src.replace("def train_loop(", WRAPPER+"def train_loop(")
    src=src.replace(anchor, OVERRIDE+anchor)
    open(f,"w").write(src)
    import py_compile; py_compile.compile(f, doraise=True)
    print("  patched model files:", n, "| run_lp.py patched + compiles")
    return n
'''))

cells.append(code(r'''
@stage("03_data_ckpt")
def _():
    cands=[p for p in glob.glob(f"/kaggle/input/**/{DATASET}", recursive=True)
           if os.path.exists(os.path.join(p, "gnn_feature"))]
    assert cands, f"{DATASET} data not attached"
    root=os.path.dirname(sorted(cands,key=len)[0])
    dst=os.path.join(REPO,"data"); os.makedirs(dst,exist_ok=True)
    for sub in os.listdir(root):
        s,d=os.path.join(root,sub),os.path.join(dst,sub)
        if os.path.isdir(s) and not os.path.exists(d): shutil.copytree(s,d)
    assert os.path.exists(os.path.join(dst,DATASET,"gnn_feature"))
    if HEART: assert os.path.isdir(os.path.join(dst,"heart",DATASET)), "HeaRT files missing"
    ck=glob.glob("/kaggle/input/**/best_model_checkpoint.pth", recursive=True)
    assert ck, "baseline checkpoint not attached"
    global CKPT; CKPT=ck[0]
    print("  data:", sorted(os.listdir(dst)), "| ckpt:", CKPT, f"{os.path.getsize(CKPT)/2**20:.0f}MB")
    return CKPT
'''))

cells.append(code((r'''
GRID = __GRID__
SEEDS = __SEEDS__

def run_one(dn, seed):
    import yaml
    os.chdir(REPO)
    c=yaml.safe_load(open(f"args/official/{DATASET}_bert.yaml"))
    if HEART: c["dataset"]="heart-"+DATASET   # same trained model, HeaRT negatives at eval
    c.update(batch_size_training=BS, gradient_accumulation_steps=32, num_workers=4,
             only_eval=True, eval_test=True, load_model_path=CKPT, seed=seed,
             save_path="/tmp/evalckpts")
    tag=f"d{dn[0][0]}_f{dn[0][1]}_s{seed}"
    c["name"]=f"eval_{tag}"
    cfg=f"args/official/eval_{tag}.yaml"; yaml.safe_dump(c, open(cfg,"w"), sort_keys=False)
    env=dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true", OMP_NUM_THREADS="2",
             TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
             EVAL_DEPTH_NEIGHBORS=json.dumps(dn), DUMP_DIR=os.path.join(OUT,tag))
    t=time.time()
    rc,tail=sh(f"torchrun --standalone --nproc_per_node=1 --nnodes=1 run_lp.py {cfg}", env=env,
               timeout=5400 if HEART else 1800)
    ok=os.path.exists(os.path.join(OUT,tag,"test.pt"))
    print(f"  [{tag}] rc={rc} dumped={ok} {time.time()-t:.0f}s")
    if not ok: print(tail)
    return ok

@stage("04_sanity_full_budget")
def _():
    import torch
    assert run_one(GRID[0], 0), "sanity run failed"
    d=torch.load(os.path.join(OUT,f"d{GRID[0][0][0]}_f{GRID[0][0][1]}_s0","test.pt"), weights_only=False)
    print("  test metrics:", d["metrics"])
    return {k: float(v) for k, v in (d["metrics"] or {}).items()}

@stage("05_sweep")
def _():
    done={}
    for dn in GRID:
        for s in SEEDS:
            if dn==GRID[0] and s==0: continue
            done[f"{dn}-{s}"]=run_one(dn, s)
    return sum(done.values()), len(done)
''').replace("__GRID__", GRID).replace("__SEEDS__", SEEDS)))

cells.append(code(r'''
@stage("06_pack")
def _():
    shutil.make_archive("/kaggle/working/evals_pack", "zip", OUT)
    return f"{os.path.getsize('/kaggle/working/evals_pack.zip')/2**20:.1f}MB"

print("\n"+"#"*70); print("# EVAL SUMMARY"); print("#"*70)
for k,(s,v) in STATE.items(): print(f"  {k:24s} {s:5s}  {str(v)[:200]}")
print(f"\ntotal runtime: {(time.time()-T0)/60:.1f} min")
'''))

nb={"cells":cells,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"},"accelerator":"GPU"},
    "nbformat":4,"nbformat_minor":4}

outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "kernel_eval" + SLUG.split("budget-eval")[1].replace("-", "_"))
os.makedirs(outdir,exist_ok=True)
json.dump(nb, open(os.path.join(outdir,"eval_kernel.ipynb"),"w",encoding="utf-8"), indent=1)
json.dump({"id":SLUG,"title":SLUG.split("/")[1],
           "code_file":"eval_kernel.ipynb","language":"python","kernel_type":"notebook",
           "is_private":True,"enable_gpu":True,"enable_internet":True,
           "dataset_sources":[DATA_SRC],
           "kernel_sources":[CKPT_KERNEL],
           "competition_sources":[]},
          open(os.path.join(outdir,"kernel-metadata.json"),"w"), indent=2)
print("wrote", outdir, len(cells), "cells")
