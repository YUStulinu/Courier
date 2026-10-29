"""Fine-tunes the wake word detector on one voice: yours, from `record`, or (with --experiment) the
test voices it misses most, to measure what personalizing does.

    python -m courier_wake.personalize --voice data/me      # -> models/wake/personal.onnx
    python -m courier_wake.personalize --experiment         # -> results/personalize.json

The model keeps learning from its training data at the same time (a replay of 30,000 training
windows), so that it learns the voice without forgetting everyone else, and the threshold is checked
again on the validation speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import dataset as D
from .evaluate import false_wakes, in_context, recall
from .model import WakeNet, export
from .synth import SYNTH, load
from .train import FEAT, OUT, choose_threshold, fit

ROOT = Path(__file__).resolve().parents[2]


def tune(base: Path, phrases: list[np.ndarray], others: list[np.ndarray], seed: int, epochs: int = 3):
    """The base model fine-tuned on these phrases (positives) and other speech of the same voice."""
    rng = np.random.default_rng(seed)
    neg_tts, _ = load(SYNTH / "train_neg.npz")
    context = others + neg_tts[:500]
    pos = D.make_windows("pos", phrases, context, 40 * len(phrases), rng)
    neg = D.make_windows("speech", others, context, 15 * len(others), rng)
    bank = D.noise_bank(D.libri_audio(D.libri_speakers("dev-clean")[:8], limit_s=600), rng, n=80)
    Xp = D.features(pos, bank, seed=seed)
    Xn = D.features(neg, bank, seed=seed + 1)
    z = np.load(FEAT / "windows-0.npz")
    X, y, Xv, yv = z["X"], z["y"], z["Xv"], z["yv"]
    pick = rng.choice(len(X), 30000, replace=False)
    Xt = np.concatenate([X[pick], Xp, Xn])
    yt = np.concatenate([y[pick], np.ones(len(Xp), np.float32), np.zeros(len(Xn), np.float32)])
    model = WakeNet()
    model.load_state_dict(torch.load(base.with_suffix(".pt"), map_location="cpu"))
    log: list = []
    return fit(model, Xt, yt, Xv, yv, epochs, 3e-4, seed, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", help="a folder of wake_*.wav and other_*.wav (from courier_wake.record)")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--base", default=str(OUT / "hey_courier.onnx"))
    args = ap.parse_args()
    base = Path(args.base)
    if args.experiment:
        experiment(base)
        return
    if not args.voice:
        ap.error("--voice or --experiment")
    d = Path(args.voice)
    phrases = [sf.read(p, dtype="float32")[0] for p in sorted(d.glob("wake_*.wav"))]
    others = [sf.read(p, dtype="float32")[0] for p in sorted(d.glob("other_*.wav"))]
    if len(phrases) < 5:
        raise SystemExit(f"{d} has {len(phrases)} recordings of the phrase; record at least 5")
    print(f"{len(phrases)} phrases, {len(others)} other sentences")
    model = tune(base, phrases, others, seed=0)
    chosen = choose_threshold(model, np.random.default_rng(0))
    out = OUT / "personal.onnx"
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    export(model, out)
    info = json.loads(base.with_suffix(".json").read_text())
    info.update({"threshold": chosen["threshold"], "validation": chosen, "personalized_from": base.name,
                 "personal_phrases": len(phrases), "personal_sentences": len(others)})
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out}; set [models] wake = \"models/wake/personal.onnx\" in courier.toml")


def experiment(base: Path, speakers: int = 10, enroll: int = 8, test: int = 24):
    """The test speakers the detector misses most, each personalized on 8 of their phrases and 8 of
    their sentences, then tested on 24 other phrases (and false wakes on an hour of test-clean)."""
    from .engine import Detector, Voice
    from . import texts

    res = json.loads((ROOT / "results" / "wake.json").read_text())
    per = res["per_speaker_recall"]
    worst = sorted(per, key=lambda k: per[k])[:speakers]
    info = json.loads(base.with_suffix(".json").read_text())
    thr = info["threshold"]
    libri = np.concatenate(D.libri_audio(D.libri_speakers("test-clean"), limit_s=3600))
    hours = len(libri) / 16000 / 3600
    base_det = Detector(base, thr)
    base_fa = false_wakes(base_det, libri)
    rows = []
    voices: dict = {}
    for i, key in enumerate(worst):
        vname, spk = key.split("#")
        spk = int(spk)
        if vname not in voices:
            voices[vname] = Voice(vname)
        v = voices[vname]
        rng = np.random.default_rng(1000 + i)

        def say(text):
            from .audio import pcm_to_float, resample, trim

            return trim(resample(pcm_to_float(v.speak(text, spk, float(rng.uniform(0.85, 1.2)), float(rng.uniform(0.6, 1.2)))), v.rate))

        phrases = [say(texts.wake_phrase(rng)) for _ in range(enroll + test)]
        others = [say(texts.COMMANDS[j % len(texts.COMMANDS)]) for j in range(enroll)]
        cases = [in_context(rng, p, others) for p in phrases[enroll:]]
        before = {"clean": recall(base_det, cases, False, 1)["recall"], "noisy": recall(base_det, cases, True, 2)["recall"]}
        model = tune(base, phrases[:enroll], others, seed=i)
        tmp = ROOT / "models" / "wake" / "_experiment.onnx"
        export(model, tmp)
        det = Detector(tmp, thr)
        after = {"clean": recall(det, cases, False, 1)["recall"], "noisy": recall(det, cases, True, 2)["recall"]}
        fa = false_wakes(det, libri)
        rows.append({"speaker": key, "base_recall_in_eval": per[key], "before": before, "after": after, "false_wakes_per_hour": fa / hours})
        print(f"{key:<32} before {before['clean']:.2f}/{before['noisy']:.2f}  after {after['clean']:.2f}/{after['noisy']:.2f}  false wakes {fa / hours:.2f}/h", flush=True)
        tmp.unlink(missing_ok=True)
    mean = lambda k1, k2: float(np.mean([r[k1][k2] for r in rows]))
    out = {
        "speakers": rows, "enroll_phrases": enroll, "test_phrases": test, "threshold": thr,
        "base_false_wakes_per_hour": base_fa / hours, "libri_hours": hours,
        "mean": {"before_clean": mean("before", "clean"), "before_noisy": mean("before", "noisy"),
                 "after_clean": mean("after", "clean"), "after_noisy": mean("after", "noisy"),
                 "false_wakes_per_hour_after": float(np.mean([r["false_wakes_per_hour"] for r in rows]))},
    }
    print(json.dumps(out["mean"], indent=2))
    (ROOT / "results" / "personalize.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
