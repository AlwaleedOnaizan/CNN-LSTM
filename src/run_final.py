"""Step 5 - train the final models with the best hyper-parameters and evaluate on test.

    python src/run_final.py --epochs 8 --seeds 42
"""
import argparse
import json

from config import OUT_DIR
from train import get_args, run

MODELS = ["cnn_avg", "cnn_lstm", "cnn_bilstm", "cnn_gru"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    best = json.loads((OUT_DIR / "best_params.json").read_text())
    for seed in a.seeds:
        for m in a.models:
            cfg = {**best, "model": m}
            argv = [f"--{k}={v}" for k, v in cfg.items()] + [f"--epochs={a.epochs}", f"--seed={seed}",
                                                              f"--tag=final_{m}_s{seed}"]
            run(get_args(argv))


if __name__ == "__main__":
    main()
