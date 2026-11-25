"""Step 4 - one-factor-at-a-time hyper-parameter search for the CNN-LSTM.

Selection is done on the identity-disjoint VALIDATION AUC only (test set untouched).
Each factor is varied while the others stay at their current best value; the best
value is then frozen before the next factor is searched. A temporal-length ablation
(number of frames T) is also run for the CNN-only baseline and the CNN-LSTM -
this plays the role of the "feature selection" study in the report template.

Results: outputs/hparam_results.csv, outputs/best_params.json
"""
import json
import pandas as pd

from config import OUT_DIR
from train import get_args, run

EPOCHS = 4
SEARCH = [                                  # (parameter, values) in search order
    ("lr_cnn", [3e-5, 1e-4, 3e-4]),
    ("hidden", [128, 256, 512]),
    ("layers", [1, 2]),
    ("dropout", [0.1, 0.3, 0.5]),
]
FRAMES = [4, 8, 12, 16, 20]
CSV = OUT_DIR / "hparam_results.csv"


def done():
    return pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()


def evaluate(cfg, study):
    tag = "hp_{model}_T{frames}_lr{lr_cnn:g}_h{hidden}_l{layers}_d{dropout:g}".format(**cfg)
    prev = done()
    if len(prev) and (prev.tag == tag).any():            # resume support
        return float(prev[prev.tag == tag].val_auc.iloc[0])
    argv = [f"--{k}={v}" for k, v in cfg.items()] + [f"--epochs={EPOCHS}", "--no_test", f"--tag={tag}"]
    r = run(get_args(argv))
    row = dict(study=study, tag=tag, **cfg, val_auc=r["best_val_auc"], val_acc=r["val"]["accuracy"],
               val_f1=r["val"]["f1"], minutes=r["train_minutes"])
    pd.concat([prev, pd.DataFrame([row])]).to_csv(CSV, index=False)
    return r["best_val_auc"]


def main():
    best = dict(model="cnn_lstm", frames=16, lr_cnn=1e-4, hidden=256, layers=1, dropout=0.3)
    for name, values in SEARCH:
        scores = {v: evaluate({**best, name: v}, name) for v in values}
        best[name] = max(scores, key=scores.get)
        print(f"== {name}: {scores} -> best {best[name]}")
    (OUT_DIR / "best_params.json").write_text(json.dumps(best, indent=2))

    for model in ["cnn_lstm", "cnn_avg"]:                # temporal ablation
        for t in FRAMES:
            evaluate({**best, "model": model, "frames": t}, f"frames_{model}")
    print("Best parameters:", best)


if __name__ == "__main__":
    main()
