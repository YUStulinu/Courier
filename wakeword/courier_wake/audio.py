"""Audio helpers: resampling, trimming, and the augmentations that make synthetic speech sound like a
microphone in a room."""

from __future__ import annotations

import numpy as np

RATE = 16000


def resample(x: np.ndarray, src: int, dst: int = RATE) -> np.ndarray:
    """Band-limited resampling through the spectrum (whole clips; padded so the circular wrap is silent)."""
    x = np.asarray(x, dtype=np.float32)
    if src == dst or len(x) == 0:
        return x.copy()
    pad = src // 10
    xp = np.concatenate([x, np.zeros(pad, np.float32)])
    n_out = int(round(len(xp) * dst / src))
    X = np.fft.rfft(xp)
    keep = n_out // 2 + 1
    if keep <= len(X):
        Y = X[:keep]
    else:
        Y = np.concatenate([X, np.zeros(keep - len(X), X.dtype)])
    y = np.fft.irfft(Y, n_out) * (n_out / len(xp))
    return y[: int(round(len(x) * dst / src))].astype(np.float32)


def pcm_to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def trim(x: np.ndarray, threshold_db: float = -40.0, pad: int = 800) -> np.ndarray:
    """Drops leading and trailing silence (relative to the clip's peak), keeping `pad` samples."""
    if len(x) == 0:
        return x
    frame = 160
    n = len(x) // frame
    if n == 0:
        return x
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    db = 10 * np.log10(e + 1e-12)
    loud = np.where(db > db.max() + threshold_db)[0]
    if len(loud) == 0:
        return x[:0]
    a = max(0, loud[0] * frame - pad)
    b = min(len(x), (loud[-1] + 1) * frame + pad)
    return x[a:b]


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def set_level(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** ((db - rms_db(x)) / 20)).astype(np.float32)


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[: len(x)]
    return y.astype(np.float32)


def room_impulse(rng: np.random.Generator, rt60: float) -> np.ndarray:
    """A synthetic room: the direct sound, then exponentially decaying noise (reflections) that
    falls 60 dB in rt60 seconds and loses its highs faster."""
    n = int(rt60 * RATE)
    t = np.arange(n) / RATE
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    # Darken the tail: a one-pole lowpass whose cutoff falls over time is approximated by blending
    # with a smoothed copy.
    smooth = np.convolve(tail, np.ones(4) / 4, mode="same")
    w = np.clip(t / rt60, 0, 1)
    tail = (1 - w) * tail + w * smooth
    delay = int(rng.uniform(0.002, 0.015) * RATE)
    h = np.zeros(n + delay, np.float32)
    h[0] = 1.0
    drr = 10 ** (rng.uniform(-3, 9) / 20)  # direct-to-reverberant ratio
    tail = tail / (np.sqrt(np.sum(tail**2)) + 1e-9) / drr
    h[delay:] += tail.astype(np.float32)
    return h


def colored_noise(rng: np.random.Generator, n: int, exponent: float) -> np.ndarray:
    """Noise with power ~ 1/f^exponent (0 white, 1 pink, 2 brown)."""
    X = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1.0
    X = X / f ** (exponent / 2)
    y = np.fft.irfft(X, n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def hum(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mains hum or a fan: a low tone and its harmonics."""
    f0 = rng.choice([50.0, 60.0, rng.uniform(80, 300)])
    t = np.arange(n) / RATE
    y = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k in range(1, 6))
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def tones(rng: np.random.Generator, n: int) -> np.ndarray:
    """Music-like: a few notes with harmonics, changing every few hundred ms."""
    y = np.zeros(n, np.float32)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.15, 0.6) * RATE)
        t = np.arange(min(length, n - pos)) / RATE
        note = np.zeros_like(t)
        for _ in range(rng.integers(1, 4)):
            f = 110 * 2 ** (rng.integers(0, 36) / 12)
            note += sum(0.5**k * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        note *= np.exp(-t * rng.uniform(1, 6))
        y[pos : pos + len(t)] += note.astype(np.float32)
        pos += length
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def equalize(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """A random smooth frequency response: a microphone and a room colour the sound."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.linspace(0, 1, len(X))
    gains_db = rng.uniform(-8, 8, 6)
    curve = np.interp(f, np.linspace(0, 1, 6), gains_db)
    # Small built-in microphones lose the lows.
    if rng.random() < 0.4:
        curve += np.where(f < 0.03, -12 * (1 - f / 0.03), 0)
    return np.fft.irfft(X * 10 ** (curve / 20), n).astype(np.float32)


def band_limit(x: np.ndarray, top_hz: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    X[f > top_hz] = 0
    return np.fft.irfft(X, len(x)).astype(np.float32)


class Augmenter:
    """Applies a random chain to clean speech: level, room, microphone, noise."""

    def __init__(self, noise_bank: list[np.ndarray] | None = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.noise_bank = noise_bank or []

    def noise(self, n: int) -> np.ndarray:
        r = self.rng.random()
        if self.noise_bank and r < 0.45:
            src = self.noise_bank[self.rng.integers(len(self.noise_bank))]
            if len(src) >= n:
                start = self.rng.integers(0, len(src) - n + 1)
                y = src[start : start + n].copy()
            else:
                y = np.resize(src, n).copy()
            return y / (np.std(y) + 1e-9)
        if r < 0.75:
            return colored_noise(self.rng, n, self.rng.uniform(0, 2))
        if r < 0.88:
            return hum(self.rng, n) + 0.3 * colored_noise(self.rng, n, 1.0)
        return tones(self.rng, n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = self.rng
        y = x.astype(np.float32)
        if rng.random() < 0.5:
            y = fft_convolve(y, room_impulse(rng, rng.uniform(0.15, 0.9)))
        if rng.random() < 0.5:
            y = equalize(rng, y)
        if rng.random() < 0.08:
            y = band_limit(y, rng.uniform(3400, 5000))
        level = rng.uniform(-42, -18)
        if rms_db(y) > -90:
            y = set_level(y, level)
        if rng.random() < 0.8:
            snr = rng.uniform(0, 30)
            noise = self.noise(len(y))
            y = y + noise * 10 ** ((level - snr) / 20)
        return np.clip(y, -1, 1).astype(np.float32)
