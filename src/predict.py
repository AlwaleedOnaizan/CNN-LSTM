"""Inference on any video file with a trained checkpoint.

    python src/predict.py path/to/video.mp4 --ckpt outputs/checkpoints/final_cnn_lstm_s42.pt
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

import extract_faces as ef
from config import RUNS_DIR, N_FRAMES_STORED
from data import MEAN, STD
from models import DeepfakeDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    tag = a.ckpt.replace("\\", "/").split("/")[-1].replace(".pt", "")
    cfg = json.loads((RUNS_DIR / tag / "config.json").read_text())

    t0 = time.time()
    ef._init()
    frames, _ = ef.read_frames(a.video, N_FRAMES_STORED)
    if not frames:
        sys.exit("Could not read video")
    boxes = [ef.detect(f) for f in frames]
    found = [i for i, b in enumerate(boxes) if b is not None]
    import cv2

    def box_for(i, f):
        if boxes[i] is not None:
            return boxes[i]
        if found:
            return boxes[min(found, key=lambda k: abs(k - i))]
        h, w = f.shape[:2]                      # no face anywhere: centre crop
        s = min(h, w) / ef.FACE_MARGIN
        return np.array([(w - s) / 2, (h - s) / 2, s, s])

    if not found:
        print("[warn] no face detected - using centre crop")
    crops = [cv2.cvtColor(ef.crop(f, box_for(i, f)), cv2.COLOR_BGR2RGB) for i, f in enumerate(frames)]
    idx = np.round(np.linspace(0, len(crops) - 1, cfg["frames"])).astype(int)
    x = (np.stack([crops[i] for i in idx]).astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).unsqueeze(0).cuda()

    model = DeepfakeDetector(cfg["model"], cfg["hidden"], cfg["layers"], cfg["dropout"]).cuda().eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p = torch.sigmoid(model(x)).item()
    print(f"P(fake) = {p:.4f}  ->  {'FAKE' if p >= 0.5 else 'REAL'}   ({time.time() - t0:.2f}s, "
          f"faces found in {len(found)}/{len(frames)} frames)")


if __name__ == "__main__":
    main()
