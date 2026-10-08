import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from embed import VGG16Pool5, DINOv2

FIXTURES = ROOT / "model" / "validate" / "fixtures"
PNG_DIR = ROOT / "model" / ".cache" / "224"
CACHE = ROOT / "model" / ".cache"


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def topk(matrix, k):
    sim = unit(matrix) @ unit(matrix).T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def overlap(a, b, k):
    ta, tb = topk(a, k), topk(b, k)
    return float(np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(len(ta))]))


def feats_for(embedder, images):
    key = embedder.name + (f"_{embedder.mode}" if hasattr(embedder, "mode") else "")
    cache = CACHE / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.concatenate([embedder.embed_batch(images[s:s + 16]) for s in range(0, len(images), 16)])
    np.save(cache, out)
    return out


def main():
    names = (FIXTURES / "metadata.tsv").read_text().strip().split("\n")
    n = len(names) - 1
    by_index = {int(p.name.split("-")[0]): p for p in PNG_DIR.glob("*.png")}
    images = [np.asarray(Image.open(by_index[i]).convert("RGB")) for i in range(n)]
    theirs = np.loadtxt(FIXTURES / "vectors-200.tsv")

    embedders = [VGG16Pool5(mode="raw01"), DINOv2()]
    print(f"{n} instances; challenger comparison (PCA 200) vs fontjoy vectors-200.tsv")
    print(f"{'backbone':<16}{'raw_dim':>9}" + "".join(f"{'ovl@' + str(k):>9}" for k in (1, 5, 10, 20)) + "     pearson")
    for emb in embedders:
        feats = feats_for(emb, images)
        ours = PCA(n_components=200, random_state=0).fit_transform(feats)
        row = [overlap(ours, theirs, k) for k in (1, 5, 10, 20)]
        idx = np.random.default_rng(0).choice(n, size=400, replace=False)
        a, b = unit(ours), unit(theirs)
        r = np.corrcoef((a[idx] @ a[idx].T)[np.triu_indices(400, 1)],
                        (b[idx] @ b[idx].T)[np.triu_indices(400, 1)])[0, 1]
        print(f"{emb.name:<16}{feats.shape[1]:>9}" + "".join(f"{v:>9.3f}" for v in row) + f"     {r:.3f}")


if __name__ == "__main__":
    main()
