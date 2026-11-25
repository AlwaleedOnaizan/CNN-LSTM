"""Central configuration for the Celeb-DF CNN-LSTM deepfake detection project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("DF_WORK_DIR", ROOT))  # override only for smoke tests
DATA_DIR = Path(os.environ.get("DF_DATA_DIR", ROOT.parent / "archive"))  # raw dataset (Kaggle zip)
FACES_DIR = WORK / "data" / "faces"           # cached face crops, one .npy per video
OUT_DIR = WORK / "outputs"                    # metrics, figures, checkpoints
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = OUT_DIR / "runs"
META_CSV = OUT_DIR / "metadata.csv"
VIDEO_STATS_CSV = OUT_DIR / "video_stats.csv"
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

SEED = 42
N_FRAMES_STORED = 20      # frames sampled uniformly per video and cached
IMG_SIZE = 224            # face crop resolution fed to the CNN
FACE_MARGIN = 1.3         # crop = 1.3 x the detected face box (keeps blending boundary)
N_FAKE_TRAIN = 1500       # fake videos sampled for training (all real videos are kept)
VAL_FRACTION = 0.15       # identity-disjoint validation split taken from the training pool

for d in (OUT_DIR, CKPT_DIR, FIG_DIR, RUNS_DIR, FACES_DIR):
    d.mkdir(parents=True, exist_ok=True)
