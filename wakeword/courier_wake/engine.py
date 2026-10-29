"""The C++ engine through ctypes: the same speech features, voice and detector the assistant uses.

Training on features computed by the engine itself means the model never meets features made any
other way at run time.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "third_party" / "models"
WAKE = ROOT / "models" / "wake"
DIM = 96  # one speech embedding every 80 ms
EMBEDDINGS = 16  # the classifier's input: the last 16
HOP = 1280  # samples between embeddings
# A fresh feature stream gives its 16th embedding after this much audio (mel frames come in blocks
# of 8: 1632 samples, then every 1280); the last embedding's audio ends 640 samples before that.
WINDOW_SAMPLES = 1632 + 24 * HOP  # 32352: 2.02 s
WINDOW_INFO_END = WINDOW_SAMPLES - 640


def _library_path() -> Path:
    name = "courier_engine.dll" if sys.platform == "win32" else "libcourier_engine.so"
    env = os.environ.get("COURIER_ENGINE")
    candidates = [Path(env)] if env else []
    candidates += [ROOT / "engine" / d / name for d in ("build-cuda", "build/Release", "build")]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"{name} not found: build engine/ first or set COURIER_ENGINE")


_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    path = _library_path()
    if sys.platform == "win32":
        os.add_dll_directory(str(path.parent))
        # A CUDA build needs the CUDA runtime (CUDA 13 keeps its DLLs in bin\x64).
        cuda = os.environ.get("CUDA_PATH")
        for sub in ("bin/x64", "bin"):
            if cuda and (Path(cuda) / sub).is_dir():
                os.add_dll_directory(str(Path(cuda) / sub))
    L = C.CDLL(str(path))
    f32p = C.POINTER(C.c_float)
    L.courier_last_error.restype = C.c_char_p
    L.courier_free_string.argtypes = [C.c_void_p]
    L.courier_free_pcm.argtypes = [C.c_void_p]
    L.courier_features_new.argtypes = [C.c_char_p]
    L.courier_features_new.restype = C.c_void_p
    L.courier_features_process.argtypes = [C.c_void_p, f32p, C.c_int, f32p, C.c_int]
    L.courier_features_process.restype = C.c_int
    L.courier_features_reset.argtypes = [C.c_void_p]
    L.courier_features_free.argtypes = [C.c_void_p]
    L.courier_tts_new.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_int]
    L.courier_tts_new.restype = C.c_void_p
    L.courier_tts_rate.argtypes = [C.c_void_p]
    L.courier_tts_speakers.argtypes = [C.c_void_p]
    L.courier_tts_speak.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float, C.POINTER(C.POINTER(C.c_int16))]
    L.courier_tts_speak.restype = C.c_int
    L.courier_tts_explain.argtypes = [C.c_void_p, C.c_char_p]
    L.courier_tts_explain.restype = C.c_void_p
    L.courier_tts_free.argtypes = [C.c_void_p]
    L.courier_wake_new.argtypes = [C.c_char_p, C.c_float, C.c_int, C.c_int]
    L.courier_wake_new.restype = C.c_void_p
    L.courier_wake_process.argtypes = [C.c_void_p, f32p, C.c_int, f32p, C.c_int, C.POINTER(C.c_int)]
    L.courier_wake_process.restype = C.c_int
    L.courier_wake_last_ms.argtypes = [C.c_void_p]
    L.courier_wake_last_ms.restype = C.c_double
    L.courier_wake_set_threshold.argtypes = [C.c_void_p, C.c_float]
    L.courier_wake_reset.argtypes = [C.c_void_p]
    L.courier_wake_free.argtypes = [C.c_void_p]
    L.courier_vad_new.argtypes = [C.c_char_p]
    L.courier_vad_new.restype = C.c_void_p
    L.courier_vad_process.argtypes = [C.c_void_p, f32p, C.c_int, f32p, C.c_int]
    L.courier_vad_process.restype = C.c_int
    L.courier_vad_free.argtypes = [C.c_void_p]
    _lib = L
    return L


def _err() -> str:
    return lib().courier_last_error().decode("utf-8", "replace")


def _f32(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(C.c_float))


class Features:
    """Speech embeddings [n, 96] of 16 kHz audio, streamed exactly as the detector streams them."""

    def __init__(self, folder: Path = WAKE):
        self.h = lib().courier_features_new(str(folder).encode())
        if not self.h:
            raise RuntimeError(_err())

    def __call__(self, audio: np.ndarray, reset: bool = True) -> np.ndarray:
        L = lib()
        if reset:
            L.courier_features_reset(self.h)
        a = np.ascontiguousarray(audio, dtype=np.float32)
        out = np.empty((len(a) // HOP + 2, DIM), dtype=np.float32)
        n = L.courier_features_process(self.h, _f32(a), len(a), _f32(out), len(out))
        if n < 0:
            raise RuntimeError(_err())
        return out[:n]

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.courier_features_free(self.h)
            self.h = None


_features = None


def features(audio: np.ndarray) -> np.ndarray:
    """Embeddings of one recording, from a fresh stream."""
    global _features
    if _features is None:
        _features = Features()
    return _features(audio)


class Voice:
    """A Piper voice. Text may contain [[IPA]] for exact pronunciations."""

    def __init__(self, name: str, threads: int = 1):
        L = lib()
        onnx = MODELS / f"{name}.onnx"
        self.h = L.courier_tts_new(str(onnx).encode(), (str(onnx) + ".json").encode(), str(MODELS / "cmudict.dict").encode(), threads)
        if not self.h:
            raise RuntimeError(_err())
        self.rate = L.courier_tts_rate(self.h)
        self.speakers = L.courier_tts_speakers(self.h)
        self.name = name

    def speak(self, text: str, speaker: int = 0, speed: float = 1.0, variation: float = 1.0) -> np.ndarray:
        """16-bit PCM at the voice's rate."""
        L = lib()
        p = C.POINTER(C.c_int16)()
        n = L.courier_tts_speak(self.h, text.encode(), speaker, speed, variation, C.byref(p))
        if n < 0:
            raise RuntimeError(_err())
        out = np.ctypeslib.as_array(p, shape=(n,)).copy() if n > 0 else np.zeros(0, np.int16)
        L.courier_free_pcm(p)
        return out

    def explain(self, text: str) -> dict:
        L = lib()
        p = L.courier_tts_explain(self.h, text.encode())
        if not p:
            raise RuntimeError(_err())
        s = C.cast(p, C.c_char_p).value.decode()
        L.courier_free_string(p)
        return json.loads(s)

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.courier_tts_free(self.h)
            self.h = None


class Detector:
    """The streaming wake word detector exactly as the assistant runs it."""

    def __init__(self, model: Path, threshold: float = 0.5, hits: int = 2, refractory_ms: int = 1500):
        L = lib()
        self.h = L.courier_wake_new(str(model).encode(), threshold, hits, refractory_ms)
        if not self.h:
            raise RuntimeError(_err())

    def process(self, audio: np.ndarray, chunk: int = 1600) -> tuple[list[float], np.ndarray]:
        """Streams audio in `chunk`-sample pieces; returns detection times (ms) and every score."""
        L = lib()
        a = np.ascontiguousarray(audio, dtype=np.float32)
        times, scores = [], []
        buf = np.empty(chunk // HOP + 8, dtype=np.float32)
        n = C.c_int(0)
        for i in range(0, len(a), chunk):
            part = a[i : i + chunk]
            d = L.courier_wake_process(self.h, _f32(part), len(part), _f32(buf), len(buf), C.byref(n))
            if d < 0:
                raise RuntimeError(_err())
            scores.extend(buf[: n.value].tolist())
            if d > 0:
                times.append(L.courier_wake_last_ms(self.h))
        return times, np.array(scores, dtype=np.float32)

    def set_threshold(self, t: float):
        lib().courier_wake_set_threshold(self.h, t)

    def reset(self):
        lib().courier_wake_reset(self.h)

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.courier_wake_free(self.h)
            self.h = None


def silero_probs(audio: np.ndarray) -> np.ndarray:
    """Silero VAD speech probability per 32 ms frame."""
    L = lib()
    h = L.courier_vad_new(str(MODELS / "silero_vad-6.2.3.onnx").encode())
    if not h:
        raise RuntimeError(_err())
    a = np.ascontiguousarray(audio, dtype=np.float32)
    out = np.empty(len(a) // 512 + 2, dtype=np.float32)
    n = L.courier_vad_process(h, _f32(a), len(a), _f32(out), len(out))
    L.courier_vad_free(h)
    return out[:n]
