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

MODES = ["caffe", "rgb_mean", "imagenet", "raw255", "raw01"]
N_COMPONENTS = [200]


def load_images(n):
    by_index = {int(p.name.split("-")[0]): p for p in PNG_DIR.glob("*.png")}
    return [np.asarray(Image.open(by_index[i]).convert("RGB")) for i in range(n)]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def topk(matrix, k):
    sim = unit(matrix) @ unit(matrix).T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def overlap(a, b, k):
    ta, tb = topk(a, k), topk(b, k)
    return float(np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(len(ta))]))


def features_for(images, mode, l2=False):
    cache = CACHE / f"vgg16_pool5_{mode}{'_l2' if l2 else ''}.npy"
    if cache.exists():
        return np.load(cache)
    embedder = VGG16Pool5(mode=mode)
    out = np.concatenate([embedder.embed_batch(images[s:s + 16]) for s in range(0, len(images), 16)])
    if l2:
        out = unit(out)
    np.save(cache, out)
    return out


def main():
    names = (FIXTURES / "metadata.tsv").read_text().strip().split("\n")
    n = len(names) - 1
    images = load_images(n)
    theirs = np.loadtxt(FIXTURES / "vectors-200.tsv")

    print(f"{n} instances; preprocessing sweep for vgg16_pool5")
    print(f"{'mode':<10}{'l2':<4}" + "".join(f"{'ovl@' + str(k):>9}" for k in (1, 5, 10, 20)) + "     pearson")
    for mode in MODES:
        for l2 in (False, True):
            t0 = time.time()
            feats = features_for(images, mode, l2)
            ours = PCA(n_components=200, random_state=0).fit_transform(feats)
            row = [overlap(ours, theirs, k) for k in (1, 5, 10, 20)]
            a, b = unit(ours), unit(theirs)
            idx = np.random.default_rng(0).choice(n, size=400, replace=False)
            r = np.corrcoef((a[idx] @ a[idx].T)[np.triu_indices(400, 1)],
                            (b[idx] @ b[idx].T)[np.triu_indices(400, 1)])[0, 1]
            print(f"{mode:<10}{str(l2):<4}" + "".join(f"{v:>9.3f}" for v in row) + f"     {r:.3f}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
