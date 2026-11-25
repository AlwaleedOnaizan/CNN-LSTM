"""Evaluation metrics and statistical tests (positive class = fake)."""
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, log_loss)


def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def all_metrics(y, p, thr=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), specificity=tn / max(tn + fp, 1),
                f1=f1_score(y, yh, zero_division=0),
                balanced_acc=(tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                eer=eer(y, p) if len(set(y)) > 1 else float("nan"),
                logloss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def bootstrap_ci(y, p, fn, n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of metric fn(y, p)."""
    rng, y, p = np.random.RandomState(seed), np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def mcnemar(y, pa, pb, thr=0.5):
    """McNemar test (continuity corrected) on paired correct/incorrect decisions."""
    ca, cb = (np.asarray(pa) >= thr) == y, (np.asarray(pb) >= thr) == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return dict(b=b, c=c, chi2=chi2, p=float(stats.chi2.sf(chi2, 1)))


def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, pa, pb):
    """DeLong et al. (1988) test for the difference between two correlated AUCs."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([pa, pb])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    zstat = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), z=float(zstat),
                p=float(2 * stats.norm.sf(abs(zstat))))
