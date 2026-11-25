"""Step 6 - produce every table and figure used in the report.

Tables  -> outputs/tables/*.csv     Figures -> outputs/figures/*.png
Summary -> outputs/summary.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score

from config import OUT_DIR, FIG_DIR, RUNS_DIR, META_CSV, VIDEO_STATS_CSV, FACES_DIR
from metrics import all_metrics, bootstrap_ci, mcnemar, delong

TAB = OUT_DIR / "tables"; TAB.mkdir(exist_ok=True)
NAMES = {"cnn_avg": "CNN (temporal avg.)", "cnn_lstm": "CNN-LSTM", "cnn_bilstm": "CNN-BiLSTM", "cnn_gru": "CNN-GRU"}
COLORS = {"cnn_avg": "#7f7f7f", "cnn_lstm": "#d62728", "cnn_bilstm": "#1f77b4", "cnn_gru": "#2ca02c"}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})
summary = {}


def dataset_analysis():
    meta = pd.read_csv(META_CSV)
    counts = meta.groupby(["split", "source"]).size().unstack(fill_value=0)
    counts.to_csv(TAB / "split_counts.csv")
    summary["split_counts"] = meta[meta.split != "unused"].groupby(["split", "label"]).size().unstack().to_dict()
    summary["full_counts"] = meta.source.value_counts().to_dict()

    st = pd.read_csv(VIDEO_STATS_CSV)
    st["class"] = st.label.map({0: "Real", 1: "Fake"})
    cols = {"duration_s": "Duration (s)", "fps": "Frame rate (fps)", "total_frames": "Frames per video",
            "width": "Width (px)", "height": "Height (px)", "detect_rate": "Face detection rate"}
    rows = []
    for cls in ["Real", "Fake", "All"]:
        d = st if cls == "All" else st[st["class"] == cls]
        for c, name in cols.items():
            v = d[c].dropna()
            rows.append(dict(Class=cls, Attribute=name, Mean=v.mean(), Median=v.median(), Std=v.std(),
                             Max=v.max(), Min=v.min()))
    pd.DataFrame(rows).round(3).to_csv(TAB / "dataset_stats.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    c2 = meta[meta.split != "unused"].groupby(["split", "label"]).size().unstack().reindex(["train", "val", "test"])
    c2.columns = ["Real", "Fake"]
    c2.plot.bar(ax=ax[0], color=["#1f77b4", "#d62728"], rot=0)
    ax[0].set_title("Videos per split"); ax[0].set_ylabel("count")
    for p in ax[0].patches:
        ax[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    sns.histplot(st, x="duration_s", hue="class", bins=40, ax=ax[1], stat="density", common_norm=False,
                 palette={"Real": "#1f77b4", "Fake": "#d62728"})
    ax[1].set_title("Video duration distribution"); ax[1].set_xlabel("seconds")
    plt.savefig(FIG_DIR / "dataset_overview.png"); plt.close()

    # example face crops
    ex = {}
    for cls, lab in [("Real", 0), ("Fake", 1)]:
        vids = st[(st.label == lab) & (st.split == "test")].video.sample(3, random_state=1)
        ex[cls] = [np.load(FACES_DIR / (v.replace("/", "__").replace(".mp4", ".npy"))) for v in vids]
    fig, axes = plt.subplots(6, 6, figsize=(9, 9.3))
    for r, (cls, arrs) in enumerate([(c, a) for c in ex for a in ex[c]]):
        for k, fi in enumerate(np.linspace(0, arrs.shape[0] - 1, 6).astype(int)):
            axes[r, k].imshow(arrs[fi]); axes[r, k].axis("off")
        axes[r, 0].set_title(cls, loc="left", fontsize=9, color="#1f77b4" if cls == "Real" else "#d62728")
    plt.suptitle("Sampled face sequences (6 of 20 frames shown)", y=0.995)
    plt.tight_layout(); plt.savefig(FIG_DIR / "sample_faces.png"); plt.close()


def frames_rows(hp):
    """All runs that use the best parameters (any T) - includes the T=16 run cached from the search."""
    best = json.loads((OUT_DIR / "best_params.json").read_text())
    m = np.ones(len(hp), bool)
    for k in ["lr_cnn", "hidden", "layers", "dropout"]:
        m &= np.isclose(hp[k].astype(float), float(best[k]))
    return hp[m].drop_duplicates("tag")


def hparam_analysis():
    f = OUT_DIR / "hparam_results.csv"
    if not f.exists():
        return
    hp = pd.read_csv(f)
    hp.to_csv(TAB / "hparam_results.csv", index=False)
    studies = [s for s in ["lr_cnn", "hidden", "layers", "dropout"] if s in hp.study.values]
    fig, axes = plt.subplots(1, len(studies), figsize=(3.2 * len(studies), 3))
    labels = {"lr_cnn": "CNN learning rate", "hidden": "LSTM hidden units", "layers": "LSTM layers", "dropout": "Dropout"}
    best = json.loads((OUT_DIR / "best_params.json").read_text())
    for ax, s in zip(np.atleast_1d(axes), studies):
        m = (hp.model == "cnn_lstm") & (hp.frames == best["frames"])
        for k in ["lr_cnn", "hidden", "layers", "dropout"]:
            if k != s:
                m &= np.isclose(hp[k].astype(float), float(best[k]))
        d = hp[m].drop_duplicates("tag").sort_values(s)
        x = d[s].astype(str) if s in ("lr_cnn",) else d[s]
        ax.plot(range(len(d)) if s == "lr_cnn" else x, d.val_auc, "o-", color="#d62728")
        if s == "lr_cnn":
            ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{v:g}" for v in d[s]])
        else:
            ax.set_xticks(d[s])
        ax.set_xlabel(labels[s]); ax.set_ylabel("Validation AUC"); ax.grid(alpha=.3); ax.margins(y=0.18)
        for xi, yi in zip(range(len(d)) if s == "lr_cnn" else d[s], d.val_auc):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    plt.suptitle("CNN-LSTM hyper-parameter search (one factor at a time)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "hparam_search.png"); plt.close()

    fr = frames_rows(hp)
    if len(fr):
        fig, ax = plt.subplots(figsize=(5, 3.3))
        for m, d in fr.groupby("model"):
            d = d.sort_values("frames")
            ax.plot(d.frames, d.val_auc, "o-", label=NAMES[m], color=COLORS[m])
        ax.set_xlabel("Frames per video (T)"); ax.set_ylabel("Validation AUC"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("Effect of temporal context (number of frames)")
        plt.savefig(FIG_DIR / "frames_ablation.png"); plt.close()
        fr.pivot_table(index="frames", columns="model", values=["val_auc", "val_acc"]).round(4).to_csv(TAB / "frames_ablation.csv")
    bp = OUT_DIR / "best_params.json"
    if bp.exists():
        summary["best_params"] = json.loads(bp.read_text())


def final_analysis():
    runs = sorted(RUNS_DIR.glob("final_*"))
    runs = [r for r in runs if (r / "test_preds.csv").exists()]
    if not runs:
        return
    rows, preds, hists = [], {}, {}
    for r in runs:
        mj = json.loads((r / "metrics.json").read_text())
        m, seed = mj["model"], mj["seed"]
        tp = pd.read_csv(r / "test_preds.csv")
        preds.setdefault(m, {})[seed] = tp
        hists.setdefault(m, {})[seed] = pd.read_csv(r / "history.csv")
        rows.append(dict(model=m, seed=seed, **mj["test"], val_auc=mj["best_val_auc"], params_M=mj["params_M"],
                         train_min=mj["train_minutes"], ms_per_video=mj.get("test_ms_per_video")))
    allr = pd.DataFrame(rows)
    allr.to_csv(TAB / "final_all_runs.csv", index=False)

    main_seed = int(allr.seed.mode()[0])
    order = [m for m in NAMES if m in preds]
    table = []
    for m in order:
        tp = preds[m][main_seed] if main_seed in preds[m] else next(iter(preds[m].values()))
        y, p = tp.label.values, tp.prob.values
        met = all_metrics(y, p)
        lo, hi = bootstrap_ci(y, p, roc_auc_score)
        alo, ahi = bootstrap_ci(y, p, lambda a, b: accuracy_score(a, b >= .5))
        row = {"Model": NAMES[m], "Accuracy": met["accuracy"], "Acc 95% CI": f"[{alo:.3f}, {ahi:.3f}]",
               "Precision": met["precision"], "Recall (TPR)": met["recall"], "Specificity": met["specificity"],
               "F1": met["f1"], "AUC": met["auc"], "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "EER": met["eer"],
               "Log-loss": met["logloss"]}
        r = allr[(allr.model == m)]
        if len(r) > 1:
            row["AUC mean±std (seeds)"] = f"{r.auc.mean():.4f} ± {r.auc.std():.4f}"
            row["Acc mean±std (seeds)"] = f"{r.accuracy.mean():.4f} ± {r.accuracy.std():.4f}"
        row["Params (M)"] = r.params_M.iloc[0]; row["Inference ms/video"] = r.ms_per_video.mean()
        table.append(row)
    res = pd.DataFrame(table)
    res.round(4).to_csv(TAB / "final_results.csv", index=False)
    summary["final_results"] = res.round(4).to_dict(orient="records")

    # statistical tests: CNN-LSTM vs every other model
    if "cnn_lstm" in preds:
        ref = preds["cnn_lstm"][main_seed]
        tests = []
        for m in order:
            if m == "cnn_lstm":
                continue
            o = preds[m][main_seed].set_index("video").loc[ref.video]
            d = delong(ref.label.values, ref.prob.values, o.prob.values)
            mc = mcnemar(ref.label.values, ref.prob.values, o.prob.values)
            tests.append({"Comparison": f"CNN-LSTM vs {NAMES[m]}", "AUC CNN-LSTM": d["auc_a"], "AUC other": d["auc_b"],
                          "DeLong z": d["z"], "DeLong p": d["p"], "McNemar b (LSTM right only)": mc["b"],
                          "McNemar c (other right only)": mc["c"], "McNemar chi2": mc["chi2"], "McNemar p": mc["p"]})
        pd.DataFrame(tests).round(4).to_csv(TAB / "significance_tests.csv", index=False)
        summary["significance"] = pd.DataFrame(tests).round(4).to_dict(orient="records")

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, zoom in zip(axes, [False, True]):
        for m in order:
            tp = preds[m][main_seed]
            fpr, tpr, _ = roc_curve(tp.label, tp.prob)
            ax.plot(fpr, tpr, color=COLORS[m], lw=2 if m == "cnn_lstm" else 1.3,
                    label=f"{NAMES[m]} (AUC={roc_auc_score(tp.label, tp.prob):.3f})")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(alpha=.3)
        if zoom:
            ax.set_xlim(-0.005, 0.25); ax.set_ylim(0.75, 1.005); ax.set_title("Zoom: FPR ≤ 0.25")
        else:
            ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_title("ROC curves - Celeb-DF (v2) test set")
            ax.legend(loc="lower right", fontsize=8)
    plt.savefig(FIG_DIR / "roc_curves.png"); plt.close()

    # confusion matrices
    fig, axes = plt.subplots(1, len(order), figsize=(3.3 * len(order), 3))
    for ax, m in zip(np.atleast_1d(axes), order):
        tp = preds[m][main_seed]
        cm = confusion_matrix(tp.label, (tp.prob >= .5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        ax.set_title(NAMES[m]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrices.png"); plt.close()

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for m in order:
        h = hists[m][main_seed]
        axes[0].plot(h.epoch, h.train_loss, "-o", ms=3, color=COLORS[m], label=NAMES[m])
        axes[1].plot(h.epoch, h.val_loss, "-o", ms=3, color=COLORS[m], label=NAMES[m])
        axes[2].plot(h.epoch, h.val_auc, "-o", ms=3, color=COLORS[m], label=NAMES[m])
    for ax, t in zip(axes, ["Training loss (weighted BCE)", "Validation loss (BCE)", "Validation AUC"]):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "training_curves.png"); plt.close()

    # score distributions
    fig, axes = plt.subplots(1, len(order), figsize=(3.3 * len(order), 2.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), order):
        tp = preds[m][main_seed]
        ax.hist(tp.prob[tp.label == 0], bins=25, range=(0, 1), alpha=.6, color="#1f77b4", label="Real")
        ax.hist(tp.prob[tp.label == 1], bins=25, range=(0, 1), alpha=.6, color="#d62728", label="Fake")
        ax.axvline(.5, color="k", ls="--", lw=.8); ax.set_title(NAMES[m]); ax.set_xlabel("P(fake)")
    np.atleast_1d(axes)[0].legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "score_distributions.png"); plt.close()

    # per-source error analysis
    meta = pd.read_csv(META_CSV).set_index("video")
    err = []
    for m in order:
        tp = preds[m][main_seed].copy()
        tp["source"] = meta.loc[tp.video, "source"].values
        tp["correct"] = (tp.prob >= .5) == tp.label
        for s, d in tp.groupby("source"):
            err.append(dict(Model=NAMES[m], Source=s, N=len(d), Accuracy=d.correct.mean()))
    pd.DataFrame(err).pivot(index="Source", columns="Model", values="Accuracy").round(4).to_csv(TAB / "accuracy_by_source.csv")




def error_grid(tag="final_cnn_lstm_s42"):
    """Image grid of the test videos misclassified by the given run (real errors first)."""
    import cv2
    f = RUNS_DIR / tag / "test_preds.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    e = t[(t.prob >= .5) != t.label.astype(bool)].sort_values(["label", "prob"])
    def strip(v):
        a = np.load(FACES_DIR / (v.replace("/", "__").replace(".mp4", ".npy")))[::4]
        r = np.concatenate(list(a), 1)
        return cv2.resize(r, (r.shape[1] // 3, r.shape[0] // 3))
    cols = [[strip(v) for v in e[e.label == lab].video] for lab in (0, 1)]
    rows = cols[0] + cols[1]
    if rows:
        h, w = rows[0].shape[:2]
        n = max(len(c) for c in cols)
        blank = np.full((h, w, 3), 255, np.uint8)
        col_imgs = [np.concatenate(c + [blank] * (n - len(c)), 0) for c in cols]
        gap = np.full((n * h, 12, 3), 255, np.uint8)
        grid = np.concatenate([col_imgs[0], gap, col_imgs[1]], 1)
        cv2.imwrite(str(FIG_DIR / "errors_grid.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    dataset_analysis()
    hparam_analysis()
    final_analysis()
    error_grid()
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Tables:", sorted(p.name for p in TAB.glob("*.csv")))
    print("Figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))
