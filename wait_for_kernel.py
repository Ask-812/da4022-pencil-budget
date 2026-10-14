"""Block until a Kaggle kernel stops RUNNING, then report. Usage: python wait_for_kernel.py <slug> [max_hours]"""
import subprocess, sys, os, time, re

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "ask812/pencil-baseline"
MAX_H = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

t0 = time.time()
last = None
while time.time() - t0 < MAX_H * 3600:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", KERNEL],
                       capture_output=True, text=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    st = m.group(1) if m else "UNKNOWN"
    if st != last:
        print(f"[{(time.time()-t0)/60:6.1f} min] {st}", flush=True)
        last = st
    if st in ("COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCELLED"):
        print(f"\nFINAL: {st} after {(time.time()-t0)/60:.1f} min")
        sys.exit(0 if st == "COMPLETE" else 1)
    time.sleep(180)

print(f"\nTIMEOUT: still {last} after {MAX_H} h")
sys.exit(2)
