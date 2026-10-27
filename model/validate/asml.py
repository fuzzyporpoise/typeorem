"""ASML prototype: asymmetric similarity metric learning for pairing.

Candidate direction 2 from the literature sweep:
Jiang et al.'s ASML (arXiv:1811.08015), the learned asymmetric challenger to
the hand-built roleScore. The score is

    f(x, y) = x' G y - (x - y)' M (x - y)

with G asymmetric, M = I + L L' kept PSD, both regularized toward the identity
(gamma, Frobenius), hinge loss on positives vs sampled negatives in equal
counts (as in the paper). Inputs are the top-K standardized corpus dims: the
full 200-dim would give 2 * 200^2 parameters against ~500 positives, so K = 32
keeps the parameter count honest; the toward-identity regularizer does the
rest of the overfitting control.

Same hygiene and evaluation as reranker.py: strict canonical positives,
negatives sampled outside the acceptable partners' families, LOOCV plus a
held-out even/odd split, hit@10/50/100 and median best-partner rank against
the hand metric. Also includes the paper's other method (dual-space k-NN,
K1 = K2 = 10) as a training-free reference, and blends of the hand metric
with the z-scored ASML score ("replacement or blend" is the open question).

    model/.venv/bin/python model/validate/asml.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from metric_ab import load_corpus, build_features  # noqa: E402
from reranker import Features, build_pairs, hand_score, dsknn_score  # noqa: E402

K = 32
GAMMA = 1.0
STEPS = 300
LR = 0.03
K1 = K2 = 10
SEEDS = range(4)


def standardize(V, k=K):
    X = V[:, :k].astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return ((X - mu) / sd).astype(np.float32)


def fit_asml(X, train, F, seed, gamma=GAMMA, steps=STEPS, lr=LR):
    """Fit (L, R) on the fold's pairs; M = I + L L', G = I + R. Deterministic."""
    rng = np.random.default_rng(seed)
    xp, yp, xn, yn = [], [], [], []
    for hi, ps, excl in train:
        rows = F.pool(hi)
        allowed = np.setdiff1d(rows, np.asarray(excl))
        neg = rng.choice(allowed, size=min(len(ps), len(allowed)), replace=False)
        xp.append(np.repeat(X[hi][None, :], len(ps), axis=0))
        yp.append(X[np.asarray(ps)])
        xn.append(np.repeat(X[hi][None, :], len(neg), axis=0))
        yn.append(X[neg])
    Xp = torch.tensor(np.vstack(xp)); Yp = torch.tensor(np.vstack(yp))
    Xn = torch.tensor(np.vstack(xn)); Yn = torch.tensor(np.vstack(yn))
    torch.manual_seed(seed)
    L = torch.zeros(K, K, requires_grad=True)
    R = torch.zeros(K, K, requires_grad=True)
    opt = torch.optim.Adam([L, R], lr=lr)
    eye = torch.eye(K)
    Xall = torch.cat([Xp, Xn]); Yall = torch.cat([Yp, Yn])
    label = torch.cat([torch.ones(len(Xp)), -torch.ones(len(Xn))])
    for _ in range(steps):
        opt.zero_grad()
        M = eye + L @ L.T
        G = eye + R
        diff = Xall - Yall
        f = (Xall @ G * Yall).sum(1) - (diff @ M * diff).sum(1)
        hinge = torch.relu(1 - label * f).mean()
        reg = (gamma / 2) * ((L @ L.T).norm(p="fro") ** 2 + R.norm(p="fro") ** 2)
        (hinge + reg).backward()
        opt.step()
    return L.detach(), R.detach()


def asml_score(X, L, R, hi, rows):
    eye = torch.eye(K)
    M = eye + L @ L.T
    G = eye + R
    x = torch.tensor(X[hi][None, :])
    Y = torch.tensor(X[rows])
    diff = x - Y
    f = (x @ G * Y).sum(1) - (diff @ M * diff).sum(1)
    return f.numpy()


def main():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    F = Features(catalog, V, feats)
    _, _, pairs = build_pairs(catalog)
    X = standardize(V)
    print(f"{len(pairs)} headings; ASML on top-{K} dims, gamma={GAMMA}, steps={STEPS}; "
          f"DS-kNN K1=K2={K1}; seeds {list(SEEDS)}\n")

    def report(name, ranks):
        r = np.array(ranks)
        print(f"  {name:<26}hit@10 {int((r <= 10).sum()):>2}/{len(r)}  hit@50 {int((r <= 50).sum()):>2}/{len(r)}"
              f"  hit@100 {int((r <= 100).sum()):>2}/{len(r)}  median {int(np.median(r)):>5}")

    def summarize(name, make):
        runs = np.array([make(s) for s in SEEDS])
        h10 = (runs <= 10).sum(1); h50 = (runs <= 50).sum(1); h100 = (runs <= 100).sum(1)
        med = np.median(runs, axis=1)
        print(f"  {name:<26}hit@10 {h10.mean():4.1f} [{h10.min()}-{h10.max()}]  hit@50 {h50.mean():4.1f} [{h50.min()}-{h50.max()}]"
              f"  hit@100 {h100.mean():4.1f} [{h100.min()}-{h100.max()}]  median {np.median(med):.0f}")

    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    def eval_fold(make_score, train, test):
        return [F.rank_of(hi, ps, make_score(hi, F.pool(hi), train)) for hi, ps, _ in test]

    def asml_eval(seed, train, test):
        L, R = fit_asml(X, train, F, seed)
        return eval_fold(lambda hi, rows, _tr: asml_score(X, L, R, hi, rows), train, test)

    def blend_eval(seed, train, test, w):
        L, R = fit_asml(X, train, F, seed)
        def make(hi, rows, _tr):
            return hand_score(F, hi, rows) + w * zscore(asml_score(X, L, R, hi, rows))
        return eval_fold(make, train, test)

    def dsknn_eval(seed, train, test):
        return eval_fold(lambda hi, rows, tr: dsknn_score(V, tr, hi, rows), train, test)

    def loocv(fn, seed):
        out = []
        for j in range(len(pairs)):
            out += fn(seed, [p for k, p in enumerate(pairs) if k != j], [pairs[j]])
        return out

    def split(fn, seed):
        ev = [p for k, p in enumerate(pairs) if k % 2 == 0]
        od = [p for k, p in enumerate(pairs) if k % 2 != 0]
        return fn(seed, ev, od) + fn(seed, od, ev)

    print("references (no training):")
    report("hand metric", [F.rank_of(hi, ps, hand_score(F, hi, F.pool(hi))) for hi, ps, _ in pairs])
    summarize("DS-kNN (paper)", lambda s: loocv(dsknn_eval, s))
    print("ASML, LOOCV:")
    summarize("asml", lambda s: loocv(asml_eval, s))
    for w in (0.5, 1.0, 2.0):
        summarize(f"hand + {w}*z(asml)", lambda s, w=w: loocv(lambda seed, tr, te: blend_eval(seed, tr, te, w), s))
    print("ASML, held-out even/odd split:")
    summarize("asml", lambda s: split(asml_eval, s))
    for w in (0.5, 1.0, 2.0):
        summarize(f"hand + {w}*z(asml)", lambda s, w=w: split(lambda seed, tr, te: blend_eval(seed, tr, te, w), s))
    print("gamma sweep, even/odd split:")
    for g in (0.1, 1.0, 10.0):
        def gamma_eval(seed, train, test, g=g):
            L, R = fit_asml(X, train, F, seed, gamma=g)
            return eval_fold(lambda hi, rows, _tr: asml_score(X, L, R, hi, rows), train, test)
        summarize(f"asml gamma={g}", lambda s, ge=gamma_eval: split(ge, s))


if __name__ == "__main__":
    main()
