"""Training windows: 2 s of audio as the detector sees it, turned into 16 speech embeddings by the engine.

A positive window ends between 0 and 0.4 s after the wake phrase, which is when the streaming
detector should fire; before the phrase there is silence or other speech, after it silence or the
start of a request ("Hey Courier, set a timer ..."). Negatives are everything else: other speech
(synthetic and real), phrases that sound alike, the phrase cut short, "courier" without "hey",
and noise.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import RATE, Augmenter
from .engine import WINDOW_INFO_END as W_END
from .engine import WINDOW_SAMPLES as W

DATA = Path(__file__).resolve().parents[1] / "data"
LIBRI = DATA / "librispeech" / "LibriSpeech"


def libri_speakers(split: str) -> list[Path]:
    return sorted(p for p in (LIBRI / split).iterdir() if p.is_dir())


def libri_audio(speakers: list[Path], limit_s: float | None = None) -> list[np.ndarray]:
    """Every utterance of these speakers, 16 kHz float."""
    out, total = [], 0.0
    for spk in speakers:
        for f in sorted(spk.rglob("*.flac")):
            a, sr = sf.read(f, dtype="float32")
            assert sr == RATE
            out.append(a)
            total += len(a) / RATE
            if limit_s and total >= limit_s:
                return out
    return out


def f32(c: np.ndarray) -> np.ndarray:
    return c.astype(np.float32) / 32768.0 if c.dtype == np.int16 else c


def place_end(rng, audio: np.ndarray, end: int) -> np.ndarray:
    """The W samples ending at `end` (zero-padded on the left)."""
    start = end - W
    if start >= 0:
        return audio[start:end].copy()
    out = np.zeros(W, np.float32)
    out[-end:] = audio[:end]
    return out


def random_crop(rng, a: np.ndarray) -> np.ndarray:
    if len(a) <= W:
        out = np.zeros(W, np.float32)
        off = rng.integers(0, W - len(a) + 1)
        out[off : off + len(a)] = a
        return out
    s = rng.integers(0, len(a) - W + 1)
    return a[s : s + W].copy()


def with_context(rng, phrase: np.ndarray, others: list[np.ndarray], max_after_s: float = 0.4) -> np.ndarray:
    """A window ending 0-0.4 s after `phrase`, with something before and after it."""
    r = rng.random()
    if r < 0.45:
        before = np.zeros(int(rng.uniform(0.0, 1.0) * RATE), np.float32)
    else:
        o = f32(others[rng.integers(len(others))])
        before = np.concatenate([o[-int(rng.uniform(0.3, 1.2) * RATE) :], np.zeros(int(rng.uniform(0.05, 0.4) * RATE), np.float32)])
    after_len = int(rng.uniform(0, max_after_s) * RATE)
    if rng.random() < 0.5:
        after = np.zeros(after_len, np.float32)
    else:
        o = f32(others[rng.integers(len(others))])
        gap = np.zeros(int(rng.uniform(0.0, 0.15) * RATE), np.float32)
        after = np.concatenate([gap, o])[:after_len]
        after = np.pad(after, (0, after_len - len(after)))
    a = np.concatenate([before, phrase, after])
    # The window's last 40 ms are past what its last embedding hears.
    a = np.concatenate([a, np.zeros(W - W_END, np.float32)])
    return place_end(rng, a, len(a))


def make_windows(kind: str, clips: list[np.ndarray], others: list[np.ndarray], n: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        c = f32(clips[rng.integers(len(clips))])
        if kind == "pos" or kind == "conf":
            out.append(with_context(rng, c, others))
        elif kind == "partial":
            # The phrase stopped at 30-65 %: "hey cour-".
            cut = c[: int(len(c) * rng.uniform(0.3, 0.65))]
            out.append(with_context(rng, cut, others))
        elif kind == "tail":
            # "courier" without "hey".
            tail = c[int(len(c) * rng.uniform(0.3, 0.45)) :]
            out.append(with_context(rng, tail, others))
        elif kind == "speech":
            out.append(random_crop(rng, c))
        elif kind == "noise":
            out.append(np.zeros(W, np.float32))
        else:
            raise ValueError(kind)
    return out


_aug: Augmenter | None = None


def _init(noise_bank, seed):
    global _aug
    _aug = Augmenter(noise_bank, seed + mp.current_process().pid)


def _features(windows: list[np.ndarray]) -> np.ndarray:
    from .engine import DIM, EMBEDDINGS, features

    out = np.empty((len(windows), EMBEDDINGS, DIM), np.float16)
    for i, w in enumerate(windows):
        a = _aug(w) if _aug is not None else w
        out[i] = features(a)[-EMBEDDINGS:]
    return out


def features(windows: list[np.ndarray], noise_bank: list[np.ndarray], seed: int, workers: int = 8, augment: bool = True) -> np.ndarray:
    """Augments each window (if asked) and computes its speech embeddings, in parallel (or here, with
    workers=1)."""
    global _aug
    chunks = [windows[i : i + 512] for i in range(0, len(windows), 512)]
    if not chunks:
        return np.zeros((0, 16, 96), np.float16)
    if not augment or workers <= 1:
        _aug = Augmenter(noise_bank, seed) if augment else None
        out = np.concatenate([_features(c) for c in chunks])
        _aug = None
        return out
    with mp.Pool(workers, initializer=_init, initargs=(noise_bank, seed)) as pool:
        parts = pool.map(_features, chunks)
    return np.concatenate(parts)


def noise_bank(speech: list[np.ndarray], rng, n: int = 300) -> list[np.ndarray]:
    """Background noise made of real speech: babble of 3-6 people, 4 s each."""
    bank = []
    for _ in range(n):
        y = np.zeros(4 * RATE, np.float32)
        for _ in range(rng.integers(3, 7)):
            s = speech[rng.integers(len(speech))]
            seg = random_crop(rng, s) if len(s) > W else s
            seg = np.resize(seg, len(y))
            y += seg * rng.uniform(0.3, 1.0)
        bank.append(y / (np.std(y) + 1e-9))
    return bank
