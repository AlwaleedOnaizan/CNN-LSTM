"""Model zoo: CNN-only baseline and CNN-RNN hybrids.

All models share the same spatial encoder (EfficientNet-B0, ImageNet pre-trained),
so any difference in performance is attributable to the temporal aggregator:

* cnn_avg    - CNN features averaged over time (no temporal modelling, baseline)
* cnn_lstm   - CNN -> unidirectional LSTM -> last hidden state  (proposed)
* cnn_bilstm - CNN -> bidirectional LSTM                          (variant)
* cnn_gru    - CNN -> GRU                                         (variant)
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

FEAT_DIM = 1280


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features, self.pool = net.features, nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                       # x: (B, T, 3, H, W)
        b, t = x.shape[:2]
        f = self.pool(self.features(x.flatten(0, 1))).flatten(1)
        return f.view(b, t, -1)                 # (B, T, 1280)


class DeepfakeDetector(nn.Module):
    def __init__(self, kind="cnn_lstm", hidden=256, layers=1, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.encoder = Encoder()
        self.drop = nn.Dropout(dropout)
        if kind == "cnn_avg":
            out_dim = FEAT_DIM
        else:
            rnn = nn.GRU if kind == "cnn_gru" else nn.LSTM
            self.bi = kind == "cnn_bilstm"
            self.rnn = rnn(FEAT_DIM, hidden, num_layers=layers, batch_first=True,
                           bidirectional=self.bi, dropout=dropout if layers > 1 else 0.0)
            out_dim = hidden * (2 if self.bi else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        f = self.encoder(x)
        if self.kind == "cnn_avg":
            z = f.mean(1)
        else:
            out, h = self.rnn(self.drop(f))
            h = h[0] if isinstance(h, tuple) else h     # LSTM returns (h, c)
            z = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        return self.head(self.drop(z)).squeeze(1)       # logit, >0 => fake

    def param_groups(self, lr_cnn, lr_head):
        enc = list(self.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if id(p) not in enc_ids]
        return [{"params": enc, "lr": lr_cnn}, {"params": rest, "lr": lr_head}]
