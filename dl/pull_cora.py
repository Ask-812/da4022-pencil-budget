"""Pull ONLY the cora files out of the 7.5 GB remote data.zip via HTTP range requests."""
import sys, os, json, time
from remotezip import RemoteZip

HERE = os.path.dirname(os.path.abspath(__file__))
url = open(os.path.join(HERE, "dl_url.txt")).read().strip()
OUT = os.path.join(HERE, "cora_subset")
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
with RemoteZip(url) as z:
    names = z.namelist()
    print(f"archive entries: {len(names)}  (listed in {time.time()-t0:.0f}s)")

    tops = sorted({n.split("/")[0] for n in names})
    print("top-level:", tops[:30])

    # what dataset dirs exist?
    lvl2 = sorted({"/".join(n.split("/")[:2]) for n in names if n.count("/") >= 1})
    print("\nlevel-2 dirs (first 40):")
    for d in lvl2[:40]:
        print("   ", d)

    cora = [n for n in names if "cora" in n.lower()]
    print(f"\ncora-matching entries: {len(cora)}")
    tot = 0
    for n in cora:
        info = z.getinfo(n)
        tot += info.file_size
        print(f"    {n}  ({info.file_size/1024:.0f} KB)")
    print(f"total cora bytes: {tot/1024**2:.1f} MB")

    json.dump({"all_entries": names[:2000], "cora": cora},
              open(os.path.join(HERE, "archive_listing.json"), "w"), indent=1)

    print("\nextracting cora...")
    for n in cora:
        if n.endswith("/"):
            continue
        z.extract(n, OUT)
        print("   got", n)

print(f"\ndone in {time.time()-t0:.0f}s")
for root, dirs, files in os.walk(OUT):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {p}  {os.path.getsize(p)/1024:.0f} KB")
