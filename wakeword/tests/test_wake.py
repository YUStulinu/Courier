"""Tests of the wake word side: audio helpers, training windows, and that what is trained is what runs
(PyTorch, the ONNX export and the engine's streaming detector give the same scores)."""

from pathlib import Path

import numpy as np
import pytest

from courier_wake import dataset as D
from courier_wake import texts
from courier_wake.audio import Augmenter, resample, trim
from courier_wake.engine import WINDOW_SAMPLES, Features, features

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "wake" / "hey_courier.onnx"


def tone(hz, rate, seconds):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_resample_keeps_a_tone():
    y = resample(tone(1000, 22050, 1.0), 22050, 16000)
    assert len(y) == 16000
    ref = tone(1000, 16000, 1.0)
    assert np.max(np.abs(y[1000:15000] - ref[1000:15000])) < 0.02


def test_trim_drops_silence():
    x = np.concatenate([np.zeros(8000, np.float32), tone(300, 16000, 0.5), np.zeros(8000, np.float32)])
    y = trim(x, pad=0)
    assert abs(len(y) - 8000) <= 320


def test_embeddings_window_and_streaming():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(WINDOW_SAMPLES) * 0.05).astype(np.float32) + tone(440, 16000, 2.1)[:WINDOW_SAMPLES]
    e = features(x)
    # A training window gives exactly the classifier's 16 embeddings.
    assert e.shape == (16, 96) and np.isfinite(e).all()
    # Streaming in pieces gives the same embeddings as all at once.
    f = Features()
    parts = [f(x[i : i + 777], reset=(i == 0)) for i in range(0, len(x), 777)]
    assert np.max(np.abs(np.concatenate(parts) - e)) < 1e-4


def test_positive_windows_end_after_the_phrase():
    rng = np.random.default_rng(0)
    phrase = tone(300, 16000, 0.7)
    for _ in range(20):
        w = D.with_context(rng, phrase, [tone(200, 16000, 1.0)])
        assert len(w) == WINDOW_SAMPLES
        # The phrase ends at most 0.4 s before the window ends: some of its tone is in the last 0.45 s.
        tail = w[-int(0.45 * 16000):]
        spec = np.abs(np.fft.rfft(tail))
        assert spec[int(300 * len(tail) / 16000)] > 1.0 or spec[int(200 * len(tail) / 16000)] > 1.0


def test_augmenter_keeps_length_and_range():
    rng_audio = tone(300, 16000, 1.5)
    aug = Augmenter(seed=1)
    for _ in range(10):
        y = aug(rng_audio)
        assert len(y) == len(rng_audio)
        assert np.abs(y).max() <= 1.0


def test_wake_phrase_variants_are_ipa():
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = texts.wake_phrase(rng)
        assert p.count("[[") == 2 and p.count("]]") == 2


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model")
def test_pytorch_onnx_and_engine_agree():
    import onnxruntime as ort
    import torch

    from courier_wake.engine import Detector
    from courier_wake.model import WakeNet
    from courier_wake.stream import stream_windows
    from courier_wake.train import score

    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(16000 * 5) * 0.05).astype(np.float32)
    audio[16000:32000] += tone(220, 16000, 1.0)
    X, starts = stream_windows(audio)
    # A run's window, cut out and computed afresh, gives the same embeddings (training sees this);
    # the runs are kept in float16, hence the tolerances.
    k = len(X) // 2
    assert np.max(np.abs(features(audio[starts[k] : starts[k] + WINDOW_SAMPLES]) - X[k])) < 0.05
    net = WakeNet()
    net.load_state_dict(torch.load(MODEL.with_suffix(".pt"), map_location="cpu"))
    torch_scores = score(net, X)
    sess = ort.InferenceSession(str(MODEL))
    onnx_scores = 1 / (1 + np.exp(-sess.run(["logit"], {"emb": X.astype(np.float32)})[0]))
    det = Detector(MODEL, 0.99)
    _, engine_scores = det.process(audio, chunk=777)
    assert len(engine_scores) == len(torch_scores)
    assert np.max(np.abs(onnx_scores - torch_scores)) < 1e-4
    assert np.max(np.abs(engine_scores - torch_scores)) < 1e-2
