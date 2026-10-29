"""Makes the speech the detector learns from, with Piper voices through the engine.

Training voices: LibriTTS-R's 904 speakers, of which the last 54 are kept for validation. Test voices
are other models the detector never hears in training: VCTK (British), L2-ARCTIC (accented English),
CMU ARCTIC and six single voices.

    python -m courier_wake.synth            # all sets, into data/synth/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import texts
from .audio import pcm_to_float, resample, trim

DATA = Path(__file__).resolve().parents[1] / "data"
SYNTH = DATA / "synth"

TRAIN_VOICE = "en_US-libritts_r-medium"
VAL_FROM = 850  # LibriTTS-R speakers 850-903 are validation
TEST_VOICES = [
    "en_GB-vctk-medium", "en_US-l2arctic-medium", "en_US-arctic-medium", "en_US-joe-medium",
    "en_US-kristin-medium", "en_US-bryce-medium", "en_US-norman-medium", "en_US-john-medium", "en_GB-cori-medium",
]

_voices: dict = {}


def _voice(name):
    from .engine import Voice

    if name not in _voices:
        _voices[name] = Voice(name, threads=1)
    return _voices[name]


def _render(job):
    voice, text, speaker, speed, variation = job
    v = _voice(voice)
    pcm = v.speak(text, speaker, speed, variation)
    x = trim(resample(pcm_to_float(pcm), v.rate))
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def sentences(limit: int = 6000) -> list[str]:
    """Ordinary speech: LibriSpeech dev-clean transcripts cut into 4-12 word pieces, and requests."""
    out = list(texts.COMMANDS)
    root = DATA / "librispeech" / "LibriSpeech" / "dev-clean"
    rng = np.random.default_rng(1)
    for f in sorted(root.rglob("*.trans.txt")):
        for line in f.read_text().splitlines():
            words = line.split()[1:]
            i = 0
            while i < len(words):
                n = int(rng.integers(4, 13))
                out.append(" ".join(words[i : i + n]).lower())
                i += n
    rng.shuffle(out)
    return out[:limit]


def jobs_for(voice: str, speakers: list[int], per_speaker: dict[str, int], seed: int, pool_text: list[str]):
    rng = np.random.default_rng(seed)
    jobs = {"pos": [], "neg": [], "conf": []}
    for s in speakers:
        for kind, n in per_speaker.items():
            for _ in range(n):
                if kind == "pos":
                    text = texts.wake_phrase(rng)
                elif kind == "conf":
                    text = texts.CONFUSABLES[rng.integers(len(texts.CONFUSABLES))]
                else:
                    text = pool_text[rng.integers(len(pool_text))]
                jobs[kind].append((voice, text, s, float(rng.uniform(0.8, 1.3)), float(rng.uniform(0.5, 1.3))))
    return jobs


def save(path: Path, clips: list[np.ndarray], meta: list[dict]):
    offsets = np.cumsum([0] + [len(c) for c in clips])
    np.savez(path, audio=np.concatenate(clips) if clips else np.zeros(0, np.int16), offsets=offsets, meta=json.dumps(meta))


def load(path: Path) -> tuple[list[np.ndarray], list[dict]]:
    z = np.load(path)
    a, off = z["audio"], z["offsets"]
    clips = [a[off[i] : off[i + 1]] for i in range(len(off) - 1)]
    return clips, json.loads(str(z["meta"]))


def run(sets: dict, pool: mp.Pool):
    for name, jobs in sets.items():
        out = SYNTH / f"{name}.npz"
        if out.exists():
            print(f"{name}: exists")
            continue
        t = time.time()
        clips = pool.map(_render, jobs, chunksize=16)
        meta = [{"voice": j[0], "text": j[1], "speaker": j[2], "speed": j[3]} for j in jobs]
        keep = [i for i, c in enumerate(clips) if len(c) > 1600]
        save(out, [clips[i] for i in keep], [meta[i] for i in keep])
        secs = sum(len(clips[i]) for i in keep) / 16000
        print(f"{name}: {len(keep)} clips, {secs / 60:.1f} min of speech, {time.time() - t:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, mp.cpu_count() - 2))
    args = ap.parse_args()
    SYNTH.mkdir(parents=True, exist_ok=True)
    pool_text = sentences()
    from .engine import Voice

    sets = {}
    train = jobs_for(TRAIN_VOICE, list(range(VAL_FROM)), {"pos": 12, "neg": 10, "conf": 4}, 10, pool_text)
    val = jobs_for(TRAIN_VOICE, list(range(VAL_FROM, 904)), {"pos": 10, "neg": 10, "conf": 4}, 11, pool_text)
    for kind in ("pos", "neg", "conf"):
        sets[f"train_{kind}"] = train[kind]
        sets[f"val_{kind}"] = val[kind]

    for i, v in enumerate(TEST_VOICES):
        n = Voice(v).speakers
        # Single voices get more takes so that every voice weighs something in the results.
        per = {"pos": 8, "neg": 6, "conf": 4} if n > 1 else {"pos": 40, "neg": 30, "conf": 20}
        t = jobs_for(v, list(range(n)), per, 100 + i, pool_text)
        for kind in ("pos", "neg", "conf"):
            sets[f"test_{kind}_{v}"] = t[kind]
    with mp.Pool(args.workers) as pool:
        run(sets, pool)


if __name__ == "__main__":
    main()
