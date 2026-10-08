import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from embed import DINOv2, VGG16Pool5

CACHE = BASE / ".cache"
OUT = BASE / "out" / "spaces"


def build(embedder, name, dim=200):
    instances = json.loads((BASE / "out" / "instances.json").read_text())
    ids = [i["id"] for i in instances if (CACHE / "224" / f"{i['id']}.png").exists()]
    imgs = [np.asarray(Image.open(CACHE / "224" / f"{i}.png").convert("RGB")) for i in ids]
    print(f"{name}: embedding {len(imgs)}")
    feats = np.concatenate([embedder.embed_batch(imgs[s:s + 16]) for s in range(0, len(imgs), 16)])
    pca = PCA(n_components=dim, random_state=0).fit(feats)
    S = pca.transform(feats)
    C = pca.components_
    for k in range(C.shape[0]):
        j = int(np.argmax(np.abs(C[k])))
        if C[k, j] < 0:
            C[k] *= -1
            S[:, k] *= -1
    scale = np.abs(S).max(axis=1)
    scale[scale == 0] = 1
    q = np.clip(np.round(S / scale[:, None] * 127), -127, 127).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.i8.bin").write_bytes(q.tobytes())
    (OUT / f"{name}.meta.json").write_text(json.dumps({
        "dim": dim, "count": int(q.shape[0]), "dtype": "int8",
        "scale": {"mode": "perRow", "values": [round(float(x), 6) for x in scale]},
    }))
    print(f"  wrote {name} {q.shape}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dinov2"):
        build(DINOv2(), "dinov2")
    if which in ("all", "vgg16"):
        build(VGG16Pool5(mode="raw01"), "vgg16")
