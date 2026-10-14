"""Verify the trained PENCIL checkpoint actually loads and holds real weights."""
import torch, glob, os, json

pth = glob.glob("ckpt/**/best_model_checkpoint.pth", recursive=True)[0]
print("file:", pth)
print("size: %.1f MB" % (os.path.getsize(pth) / 1024**2))

ck = torch.load(pth, map_location="cpu", weights_only=False)
print("top-level type:", type(ck).__name__)
if isinstance(ck, dict):
    keys = list(ck.keys())
    print("top-level keys:", keys[:8], "..." if len(keys) > 8 else "")

sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
tensors = {k: v for k, v in sd.items() if hasattr(v, "shape")}
print("\ntensor entries:", len(tensors))
total = sum(v.numel() for v in tensors.values())
print("total params: {:,}".format(total))
print("DDP-wrapped (module. prefix):", any(k.startswith("module.") for k in tensors))

print("\nsample tensors:")
for k in list(tensors)[:6]:
    print(f"  {k:60s} {tuple(tensors[k].shape)}")

# the components that matter for this project
for want in ("mp_proj", "input_proj", "lp_head"):
    hit = [k for k in tensors if want in k]
    print(f"\n{want}: {len(hit)} tensors")
    for k in hit[:3]:
        print("   ", k, tuple(tensors[k].shape))

# sanity: weights must not be all-zero / NaN
import math
bad = [k for k, v in tensors.items()
       if v.dtype.is_floating_point and (torch.isnan(v).any() or v.abs().sum() == 0)]
print("\nNaN or all-zero tensors:", len(bad), bad[:3])

bm = glob.glob("ckpt/**/best_metrics.json", recursive=True)
if bm:
    m = json.load(open(bm[0]))
    print("\ntest MRR from best_metrics.json: %.4f" % m["test/mrr"])
    print("paper (cora, w/o features):        0.4223")

print("\nCHECKPOINT USABLE:", len(tensors) > 0 and not bad)
