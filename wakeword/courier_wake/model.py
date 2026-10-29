"""The detector's classifier: a small network over the last 16 speech embeddings (about 2 s).

The embeddings come from Google's speech_embedding model, frozen, which was trained on a large amount
of real speech; only this classifier is trained here, on synthetic voices. (A first version learned
its own features from log-mel spectrograms with a DS-CNN; it recognised the synthesizer it was
trained on rather than the phrase: see the README.)
"""

from __future__ import annotations

import torch
from torch import nn


class WakeNet(nn.Module):
    def __init__(self, dim: int = 96, steps: int = 16, hidden: int = 128):
        super().__init__()
        # Each embedding on its own, then the sequence as a whole.
        self.frame = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(steps * hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(hidden, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb [batch, 16, 96] -> logit [batch]."""
        return self.head(self.seq(self.frame(emb))).squeeze(1)


def export(model: WakeNet, path) -> None:
    """ONNX with a dynamic batch: input "emb" [batch, 16, 96], output "logit" [batch]."""
    model = model.eval().cpu()
    x = torch.zeros(1, 16, 96)
    torch.onnx.export(
        model, (x,), str(path), input_names=["emb"], output_names=["logit"],
        dynamic_axes={"emb": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False,
    )
