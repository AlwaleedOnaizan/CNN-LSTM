"""Step 2 - sample N frames per video, detect & crop the face, cache as .npy.

Each video becomes an array of shape (N_FRAMES_STORED, IMG_SIZE, IMG_SIZE, 3) uint8 RGB.
Face detector: YuNet (OpenCV DNN). Frames where no face is found reuse the nearest
detected box (faces move little between sampled frames). Per-video statistics
(duration, fps, resolution, detection rate) are written for the dataset analysis.
"""
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (META_CSV, FACES_DIR, VIDEO_STATS_CSV, YUNET_MODEL,
                    N_FRAMES_STORED, IMG_SIZE, FACE_MARGIN)

DET_WIDTH = 640
_detector = None


def _init():
    global _detector
    cv2.setNumThreads(1)
    _detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), 0.6, 0.3, 5000)


def detect(frame):
    h, w = frame.shape[:2]
    s = DET_WIDTH / w if w > DET_WIDTH else 1.0
    small = cv2.resize(frame, (int(w * s), int(h * s))) if s != 1.0 else frame
    _detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _detector.detect(small)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3] * f[14])   # area x confidence
    x, y, bw, bh = best[:4] / s
    return np.array([x, y, bw, bh])


def crop(frame, box):
    x, y, bw, bh = box
    cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * FACE_MARGIN
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    h, w = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    face = frame[y0:y1, x0:x1]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def read_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    targets = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    frames, i = [], 0
    while len(frames) < len(targets):
        if not cap.grab():
            break
        if i in targets:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(fr)
        i += 1
    cap.release()
    return frames, dict(total_frames=total, fps=fps, width=w, height=h)


def process(row):
    video, path = row
    out = FACES_DIR / (video.replace("/", "__").replace(".mp4", ".npy"))
    frames, info = read_frames(path, N_FRAMES_STORED)
    info["video"] = video
    if not frames:
        info.update(n_sampled=0, detect_rate=0.0, ok=False)
        return info
    boxes = [detect(f) for f in frames]
    found = [i for i, b in enumerate(boxes) if b is not None]
    info.update(n_sampled=len(frames), detect_rate=len(found) / len(frames), ok=True)
    if not out.exists():
        crops = []
        for i, f in enumerate(frames):
            if boxes[i] is None:
                if found:
                    j = min(found, key=lambda k: abs(k - i))
                    box = boxes[j]
                else:   # no face at all: centre square crop
                    h, w = f.shape[:2]
                    s = min(h, w) / FACE_MARGIN
                    box = np.array([(w - s) / 2, (h - s) / 2, s, s])
            else:
                box = boxes[i]
            crops.append(cv2.cvtColor(crop(f, box), cv2.COLOR_BGR2RGB))
        while len(crops) < N_FRAMES_STORED:      # very short videos: repeat last frame
            crops.append(crops[-1])
        np.save(out, np.stack(crops).astype(np.uint8))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(META_CSV)
    meta = meta[meta.split != "unused"]
    rows = list(zip(meta.video, meta.path))
    with Pool(args.workers, initializer=_init) as pool:
        stats = list(tqdm(pool.imap_unordered(process, rows, chunksize=4), total=len(rows)))
    stats = pd.DataFrame(stats).merge(meta[["video", "label", "split", "source"]], on="video")
    stats["duration_s"] = stats.total_frames / stats.fps.replace(0, np.nan)
    stats.to_csv(VIDEO_STATS_CSV, index=False)
    print(f"Done. Failed videos: {(~stats.ok).sum()} | mean face detection rate: {stats.detect_rate.mean():.3f}")


if __name__ == "__main__":
    main()
