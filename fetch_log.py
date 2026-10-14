"""Fetch + summarise a pencil kernel log.  Usage: python fetch_log.py [kernel-slug]"""
import json, subprocess, sys, os, glob

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "ask812/pencil-spike"
SLUG = KERNEL.split("/")[-1]
OUT = "klog_" + SLUG
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

for f in glob.glob(f"{OUT}/*.log"):
    os.remove(f)

subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KERNEL,
                "-p", OUT, "--file-pattern", rf"{SLUG}\.log", "-o", "-q"],
               env=env, check=False)

logs = glob.glob(f"{OUT}/*.log")
if not logs:
    sys.exit(f"no log downloaded for {KERNEL} (still running?)")

d = json.load(open(logs[0], encoding="utf-8"))
txt = "".join(e.get("data", "") for e in d)
open(f"{OUT}/clean.txt", "w", encoding="utf-8").write(txt)
print(f"{KERNEL}: {len(d)} entries, {len(txt)} chars\n" + "=" * 72)

KEYS = ("### STAGE", "--- STAGE", "VERDICT", "SUMMARY", "BEST", "ep ", "  ep",
        "Traceback", "Error", "error:", "exit code", "total runtime",
        "valid_mrr", "eval points", "saved ")
for line in txt.splitlines():
    if any(k in line for k in KEYS):
        print(line)
print("=" * 72)
