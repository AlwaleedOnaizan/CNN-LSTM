# DeepFake Video Detection with a CNN-LSTM Hybrid (Celeb-DF v2)

ARTI 502 – Deep Learning, final project.

A spatial CNN (EfficientNet-B0, ImageNet pre-trained) encodes each face crop and an LSTM
models the temporal consistency of the sequence. The model is trained end-to-end and compared
against a CNN-only baseline (temporal average pooling), a CNN-BiLSTM and a CNN-GRU.

## Setup
```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install opencv-python numpy pandas scikit-learn scipy matplotlib seaborn tqdm
```
Hardware used: NVIDIA RTX 3060 Ti (8 GB), Windows 11, Python 3.14, PyTorch 2.11 (CUDA 12.8).

Dataset: Celeb-DF (v2) extracted to `../archive/` (folders `Celeb-real`, `Celeb-synthesis`,
`YouTube-real` and `List_of_testing_videos.txt`). Face detector: YuNet ONNX model in `models/`.

## Reproduce
```
cd src
python prepare_split.py          # official test list + identity-disjoint train/val split
python extract_faces.py          # 20 uniformly sampled frames/video -> 224x224 face crops (.npy)
python hparam_search.py          # one-factor-at-a-time search on validation AUC + frames ablation
python run_final.py --epochs 8 --seeds 42 7 2024
python analyze.py                # tables (outputs/tables) and figures (outputs/figures)
python predict.py some_video.mp4 --ckpt ../outputs/checkpoints/final_cnn_lstm_s42.pt
```
All random seeds are fixed (default 42).

## Results (official Celeb-DF v2 test set, 518 videos)

| Model | Accuracy | AUC | F1 | EER | AUC over 3 seeds |
|---|---|---|---|---|---|
| CNN only (EfficientNet-B0, temporal avg.) | 95.17% | 0.9862 | 0.9634 | 6.03% | 0.9928 ± 0.0058 |
| **CNN-LSTM (proposed)** | **96.33%** | 0.9932 | **0.9721** | **3.58%** | 0.9927 ± 0.0013 |
| CNN-BiLSTM | 95.75% | **0.9941** | 0.9673 | 4.45% | – |
| CNN-GRU | 94.40% | 0.9910 | 0.9578 | 6.18% | – |

Single-run numbers are for seed 42. The differences between models are not statistically significant
(DeLong / McNemar p > 0.05). The CNN-LSTM was about 4x more stable across seeds than the CNN-only model.
Best hyper-parameters (validation AUC): CNN lr 3e-5, LSTM 256 units, 1 layer, dropout 0.3, T = 16 frames.

## Layout
```
src/            pipeline scripts (see "Reproduce")
models/         YuNet face detector (ONNX)
outputs/        metadata, per-run metrics & predictions, tables/, figures/
data/           cached face crops (not in git)
```
