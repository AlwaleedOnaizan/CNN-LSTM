"""Step 3 - train one model end-to-end and evaluate it.

Example:
    python src/train.py --model cnn_lstm --frames 16 --hidden 256 --epochs 6 --tag final_cnn_lstm

Outputs in outputs/runs/<tag>/: config.json, history.csv, metrics.json,
val_preds.csv, test_preds.csv (+ checkpoint in outputs/checkpoints/<tag>.pt)
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import RUNS_DIR, CKPT_DIR, SEED
from data import FaceSequenceDataset, load_split
from metrics import all_metrics
from models import DeepfakeDetector


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_lstm", choices=["cnn_avg", "cnn_lstm", "cnn_bilstm", "cnn_gru"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr_cnn", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_test", action="store_true", help="skip test evaluation (used during tuning)")
    ap.add_argument("--tag", default=None)
    return ap.parse_args(argv)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(wid):
    np.random.seed((torch.initial_seed() + wid) % 2**32)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(model(x.to(dev, non_blocking=True).to(memory_format=torch.contiguous_format)).float()).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def run(args):
    set_seed(args.seed)
    tag = args.tag or f"{args.model}_T{args.frames}_h{args.hidden}_l{args.layers}_lr{args.lr_cnn:g}_s{args.seed}"
    out = RUNS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    dev = "cuda"

    tr_df, va_df = load_split("train"), load_split("val")
    mk = lambda df, train: DataLoader(FaceSequenceDataset(df, args.frames, train), batch_size=args.batch,
                                      shuffle=train, num_workers=args.workers, pin_memory=True,
                                      drop_last=train, worker_init_fn=worker_init,
                                      persistent_workers=args.workers > 0)
    tr_dl, va_dl = mk(tr_df, True), mk(va_df, False)

    model = DeepfakeDetector(args.model, args.hidden, args.layers, args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.param_groups(args.lr_cnn, args.lr_head), weight_decay=args.wd)
    steps = args.epochs * len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_cnn, args.lr_head], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    n_pos, n_neg = tr_df.label.sum(), (1 - tr_df.label).sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=dev))  # class balancing
    scaler = torch.amp.GradScaler()

    print(f"[{tag}] train={len(tr_df)} val={len(va_df)} params={n_params/1e6:.2f}M steps={steps}")
    hist, best_auc, t0 = [], -1, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tqdm(tr_dl, desc=f"ep{ep}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        yv, pv = predict(model, va_dl, dev)
        m = all_metrics(yv, pv)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(pv).clamp(1e-7, 1 - 1e-7), torch.tensor(yv)))
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=val_loss, val_acc=m["accuracy"], val_auc=m["auc"],
                         val_f1=m["f1"], minutes=(time.time() - t0) / 60))
        print(f"  ep{ep} train_loss={tot/n:.4f} val_loss={val_loss:.4f} val_acc={m['accuracy']:.4f} val_auc={m['auc']:.4f}")
        score = m["auc"] if np.isfinite(m["auc"]) else m["accuracy"]   # AUC undefined if val has one class
        if score > best_auc:
            best_auc = score
            torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
            pd.DataFrame(dict(video=va_df.video, label=yv, prob=pv)).to_csv(out / "val_preds.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    result = dict(tag=tag, **{k: v for k, v in vars(args).items() if k != "tag"}, params_M=n_params / 1e6,
                  train_minutes=(time.time() - t0) / 60, best_val_auc=best_auc)
    val_best = pd.read_csv(out / "val_preds.csv")
    result["val"] = all_metrics(val_best.label, val_best.prob)
    if not args.no_test:
        model.load_state_dict(torch.load(CKPT_DIR / f"{tag}.pt", map_location=dev))
        te_df = load_split("test")
        te_dl = mk(te_df, False)
        t1 = time.time()
        yt, pt = predict(model, te_dl, dev)
        result["test_ms_per_video"] = 1000 * (time.time() - t1) / len(te_df)
        result["test"] = all_metrics(yt, pt)
        pd.DataFrame(dict(video=te_df.video, label=yt, prob=pt)).to_csv(out / "test_preds.csv", index=False)
        print(f"  TEST acc={result['test']['accuracy']:.4f} auc={result['test']['auc']:.4f} f1={result['test']['f1']:.4f}")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    run(get_args())
