"""What the streaming detector sees over a recording, for training and evaluation. Kept free of
PyTorch so that worker processes stay small."""

from __future__ import annotations

import numpy as np

from .engine import EMBEDDINGS, HOP, WINDOW_SAMPLES, Features

_stream = None


def stream_windows(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every run of 16 embeddings the detector scores over `audio`, and where in the audio each run's
    window starts (a fresh stream started there gives the same embeddings)."""
    global _stream
    if _stream is None:
        _stream = Features()
    e = _stream(audio)
    if len(e) < EMBEDDINGS:
        return np.zeros((0, EMBEDDINGS, 96), np.float16), np.zeros(0, np.int64)
    starts = np.arange(0, len(e) - EMBEDDINGS + 1)
    idx = starts[:, None] + np.arange(EMBEDDINGS)[None, :]
    return e[idx].astype(np.float16), starts * HOP


def pieces(sources: list[np.ndarray], seconds: int = 300) -> list[np.ndarray]:
    """Long recordings cut into 5-minute pieces on the embeddings' grid, overlapping by one window so
    that every window of the whole is a window of some piece."""
    step = (seconds * 16000) // HOP * HOP
    out = []
    for a in sources:
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a
        for i in range(0, max(1, len(a) - WINDOW_SAMPLES + HOP), step):
            out.append(a[i : i + step + WINDOW_SAMPLES])
    return out
