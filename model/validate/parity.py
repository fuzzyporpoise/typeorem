import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from embed import VGG16Pool5

FIXTURES = ROOT / "model" / "validate" / "fixtures"
PNG_DIR = ROOT / "model" / ".cache" / "224"
CACHE = ROOT / "model" / ".cache"


def load_names():
    lines = (FIXTURES / "metadata.tsv").read_text().strip().split("\n")
    return [ln.split("\t")[0] for ln in lines[1:]]


def load_images(n):
    by_index = {}
    for p in PNG_DIR.glob("*.png"):
        by_index[int(p.name.split("-")[0])] = p
    return [np.asarray(Image.open(by_index[i]).convert("RGB")) for i in range(n)]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def topk(matrix, k):
    sim = unit(matrix) @ unit(matrix).T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def overlap(a, b, k):
    ta, tb = topk(a, k), topk(b, k)
    return np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(len(ta))])


def main():
    names = load_names()
    n = len(names)
    print(f"{n} instances; loading PNGs from {PNG_DIR}")
    images = load_images(n)

    embedder = VGG16Pool5()
    print(f"embedder {embedder.name} on {embedder.device}")

    cache = CACHE / f"{embedder.name}.npy"
    if cache.exists():
        feats = np.load(cache)
        print(f"loaded cached features {feats.shape}")
    else:
        batch = 16
        chunks = []
        t0 = time.time()
        for start in range(0, n, batch):
            chunks.append(embedder.embed_batch(images[start:start + batch]))
            if start % (batch * 10) == 0:
                done = min(start + batch, n)
                print(f"  {done}/{n}  {done / (time.time() - t0):.1f} img/s")
        feats = np.concatenate(chunks)
        np.save(cache, feats)
        print(f"embedded in {time.time() - t0:.1f}s, shape {feats.shape}")

    print("PCA to 200")
    ours = PCA(n_components=200, random_state=0).fit_transform(feats)
    theirs = np.loadtxt(FIXTURES / "vectors-200.tsv")

    print(f"ours {ours.shape}  theirs {theirs.shape}")
    for k in (1, 5, 10, 20):
        print(f"  neighbor overlap@{k}: {overlap(ours, theirs, k):.3f}")

    a = unit(ours)
    b = unit(theirs)
    idx = np.random.default_rng(0).choice(n, size=min(n, 400), replace=False)
    da = (a[idx] @ a[idx].T)[np.triu_indices(len(idx), 1)]
    db = (b[idx] @ b[idx].T)[np.triu_indices(len(idx), 1)]
    print(f"  cosine-similarity Pearson r on {len(da)} pairs: {np.corrcoef(da, db)[0, 1]:.3f}")

    print("\nsanity: nearest neighbor of a few fonts (ours vs theirs)")
    for name in ("Roboto regular", "Montserrat regular", "Lora regular"):
        i = names.index(name)
        ta = np.argsort(-(unit(ours) @ unit(ours)[i]))[1:5]
        tb = np.argsort(-(unit(theirs) @ unit(theirs)[i]))[1:5]
        print(f"  {name}:")
        print(f"    ours   {[names[j] for j in ta]}")
        print(f"    theirs {[names[j] for j in tb]}")


if __name__ == "__main__":
    main()
