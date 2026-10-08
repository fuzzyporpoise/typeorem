import numpy as np
from sklearn.decomposition import PCA

DIM = 200


def fit_pca(feats, dim=DIM):
    pca = PCA(n_components=dim, random_state=0).fit(feats)
    scores = pca.transform(feats)
    comps = pca.components_
    for k in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[k])))
        if comps[k, j] < 0:
            comps[k] = -comps[k]
            scores[:, k] = -scores[:, k]
    return pca, scores


def quantize_int8(scores):
    scale = np.abs(scores).max(axis=1)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(scores / scale[:, None] * 127.0), -127, 127).astype(np.int8)
    return q, scale
