"""Pull citeseer + pubmed (core split, features, HeaRT negatives) from the 7.5 GB remote data.zip.
The precomputed heuristics/ folder is skipped: training and evaluation do not read it."""
import os, time
from remotezip import RemoteZip

url = open(os.path.join(os.path.dirname(__file__), "dl_url.txt")).read().strip()
OUT = os.path.join(os.path.dirname(__file__), "planetoid_subset")
WANT = ("data/citeseer/", "data/pubmed/", "data/heart/citeseer/", "data/heart/pubmed/")

t0 = time.time()
with RemoteZip(url) as z:
    picks = [i for i in z.infolist() if i.filename.startswith(WANT) and not i.filename.endswith("/")]
    print(f"{len(picks)} files, {sum(i.file_size for i in picks)/2**20:.1f} MB")
    for i in picks:
        z.extract(i.filename, OUT)
        print(f"  {i.filename}  {i.file_size/2**20:.1f} MB", flush=True)
print(f"done in {time.time()-t0:.0f}s")
