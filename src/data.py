"""PyTorch dataset over cached face-crop sequences."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import META_CSV, FACES_DIR, N_FRAMES_STORED

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def face_file(video):
    return FACES_DIR / (video.replace("/", "__").replace(".mp4", ".npy"))


def load_split(split):
    meta = pd.read_csv(META_CSV)
    df = meta[meta.split == split].reset_index(drop=True)
    exists = df.video.map(lambda v: face_file(v).exists())
    if (~exists).any():
        print(f"[warn] {(~exists).sum()} {split} videos have no cached faces and are skipped")
    return df[exists].reset_index(drop=True)


class FaceSequenceDataset(Dataset):
    """Returns (T, 3, H, W) float tensor and label.

    Training: random temporal offset + clip-consistent augmentation (the same flip /
    colour change is applied to every frame so temporal artefacts are preserved).
    """

    def __init__(self, df, n_frames=16, train=False):
        self.videos, self.labels = df.video.tolist(), df.label.values.astype(np.float32)
        self.n_frames, self.train = n_frames, train

    def __len__(self):
        return len(self.videos)

    def _indices(self):
        if self.n_frames >= N_FRAMES_STORED:
            return np.arange(N_FRAMES_STORED)
        if self.train:   # random sorted subset spread over the clip
            base = np.linspace(0, N_FRAMES_STORED - 1, self.n_frames)
            jitter = np.random.uniform(-0.5, 0.5, self.n_frames) * (N_FRAMES_STORED / self.n_frames)
            return np.clip(np.round(base + jitter), 0, N_FRAMES_STORED - 1).astype(int)
        return np.round(np.linspace(0, N_FRAMES_STORED - 1, self.n_frames)).astype(int)

    def __getitem__(self, i):
        x = np.load(face_file(self.videos[i]), mmap_mode="r")[self._indices()].astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            b = np.random.uniform(-0.1, 0.1)
            c = np.random.uniform(0.85, 1.15)
            x = np.clip((x - 0.5) * c + 0.5 + b, 0, 1)
        x = (x - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return x, torch.tensor(self.labels[i])
