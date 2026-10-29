"""Evaluates the wake word detector on voices it never heard, through the engine's streaming detector
(the code the assistant runs).

    python -m courier_wake.evaluate [--model models/wake/hey_courier.onnx]

- Recall: each test phrase in its own short recording (silence, "Hey Courier", a request), clean and
  in a noisy room (reverberation, noise at 5-20 dB SNR). Found if the detector fires between the
  phrase's start and 0.75 s after its end; latency is from the phrase's end to the detection.
- False wakes per hour: LibriSpeech test-clean (5.4 h of read speech, 40 speakers), clean and with
  noise; and per 100 phrases that sound alike ("hey Courtney", "the courier is late") or are ordinary
  requests, said by the test voices.
- A threshold sweep (recall vs. false wakes per hour) for the plot.
Results go to results/wake.json and docs/wake.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import dataset as D
from .audio import Augmenter
from .engine import Detector
from .synth import SYNTH, TEST_VOICES, load

ROOT = Path(__file__).resolve().parents[2]


def in_context(rng, phrase: np.ndarray, requests: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Silence, the phrase, a short gap, the start of a request, silence. Returns audio, phrase start/end."""
    # The detector scores once it has 2 s of audio, as it always has while listening.
    lead = np.zeros(int(rng.uniform(2.2, 2.8) * 16000), np.float32)
    req = D.f32(requests[rng.integers(len(requests))])[: int(1.5 * 16000)]
    gap = np.zeros(int(rng.uniform(0.05, 0.25) * 16000), np.float32)
    tail = np.zeros(int(0.8 * 16000), np.float32)
    p = D.f32(phrase)
    a = np.concatenate([lead, p, gap, req, tail])
    return a, len(lead), len(lead) + len(p)


def recall(det: Detector, cases, noisy: bool, seed: int) -> dict:
    aug = Augmenter(seed=seed)
    found, lat = 0, []
    for a, s, e in cases:
        if noisy:
            a = aug(a)
        det.reset()
        times, _ = det.process(a)
        hits = [t for t in times if s / 16 <= t <= e / 16 + 750]
        if hits:
            found += 1
            lat.append(hits[0] - e / 16)
    return {"phrases": len(cases), "recall": found / max(1, len(cases)), "latency_ms_median": float(np.median(lat)) if lat else None}


def false_wakes(det: Detector, audio: np.ndarray) -> int:
    det.reset()
    times, _ = det.process(audio, chunk=16000)
    return len(times)


def _count(job):
    model, thr, piece = job
    return false_wakes(Detector(Path(model), thr), piece)


def false_wakes_parallel(model: Path, thr: float, audio: np.ndarray, workers: int = 4) -> int:
    """False wakes over a long recording, cut into ten-minute pieces counted in parallel."""
    import multiprocessing as mp

    step = 16000 * 600
    jobs = [(str(model), thr, audio[i : i + step]) for i in range(0, len(audio), step)]
    with mp.Pool(workers) as pool:
        return sum(pool.map(_count, jobs, chunksize=1))


def stream(clips: list[np.ndarray], rng) -> np.ndarray:
    parts = []
    for c in clips:
        parts += [D.f32(c), np.zeros(int(rng.uniform(0.3, 1.0) * 16000), np.float32)]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def noisy_copy(a: np.ndarray, seed: int) -> np.ndarray:
    """Long audio in a noisy room, piece by piece (the augmenter works on short clips)."""
    aug = Augmenter(seed=seed)
    step = 16000 * 20
    return np.concatenate([aug(a[i : i + step]) for i in range(0, len(a), step)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "wake" / "hey_courier.onnx"))
    ap.add_argument("--out", default=str(ROOT / "results" / "wake.json"))
    args = ap.parse_args()
    model = Path(args.model)
    info = json.loads(model.with_suffix(".json").read_text())
    thr = info["threshold"]
    det = Detector(model, thr, info.get("hits", 2), info.get("refractory_ms", 1500))
    rng = np.random.default_rng(7)
    out = {"model": model.name, "threshold": thr, "voices": {}, "false_wakes": {}}

    all_cases = []
    for v in TEST_VOICES:
        pos, pmeta = load(SYNTH / f"test_pos_{v}.npz")
        neg, _ = load(SYNTH / f"test_neg_{v}.npz")
        conf, _ = load(SYNTH / f"test_conf_{v}.npz")
        cases = [in_context(rng, p, neg) for p in pos]
        all_cases += [(v, m["speaker"], c) for m, c in zip(pmeta, cases)]
        clean = recall(det, cases, False, 1)
        noisy = recall(det, cases, True, 2)
        # Look-alikes and requests by the same voices, one long recording each.
        conf_fa = false_wakes(det, stream(conf, rng))
        neg_fa = false_wakes(det, stream(neg, rng))
        out["voices"][v] = {
            "speakers": len({m["speaker"] for m in pmeta}), "clean": clean, "noisy": noisy,
            "look_alikes": len(conf), "look_alike_wakes": conf_fa, "requests": len(neg), "request_wakes": neg_fa,
        }
        print(f"{v:<26} recall clean {clean['recall']:.3f} noisy {noisy['recall']:.3f}  "
              f"latency {clean['latency_ms_median'] or 0:.0f} ms  look-alike wakes {conf_fa}/{len(conf)}  request wakes {neg_fa}/{len(neg)}", flush=True)

    # Real speech: LibriSpeech test-clean, clean and noisy.
    libri = np.concatenate(D.libri_audio(D.libri_speakers("test-clean")))
    hours = len(libri) / 16000 / 3600
    fa_clean = false_wakes_parallel(model, thr, libri)
    fa_noisy = false_wakes_parallel(model, thr, noisy_copy(libri, 3))
    out["false_wakes"] = {"librispeech_test_clean_hours": hours, "clean": fa_clean, "clean_per_hour": fa_clean / hours,
                          "noisy": fa_noisy, "noisy_per_hour": fa_noisy / hours}
    print(f"LibriSpeech test-clean ({hours:.1f} h): {fa_clean} false wakes clean ({fa_clean / hours:.2f}/h), "
          f"{fa_noisy} noisy ({fa_noisy / hours:.2f}/h)", flush=True)

    # Totals over every test phrase.
    tot = {k: sum(x[k] for x in out["voices"].values()) for k in ("look_alikes", "look_alike_wakes", "requests", "request_wakes")}
    n = sum(x["clean"]["phrases"] for x in out["voices"].values())
    out["total"] = {
        "phrases": n,
        "recall_clean": sum(x["clean"]["recall"] * x["clean"]["phrases"] for x in out["voices"].values()) / n,
        "recall_noisy": sum(x["noisy"]["recall"] * x["noisy"]["phrases"] for x in out["voices"].values()) / n,
        **tot,
    }
    print(json.dumps(out["total"], indent=2))

    # Per speaker, for the personalization experiment: which test speakers it misses most.
    per = {}
    for (v, spk, (a, s, e)) in all_cases:
        det.reset()
        times, _ = det.process(a)
        ok = any(s / 16 <= t <= e / 16 + 750 for t in times)
        per.setdefault(f"{v}#{spk}", []).append(ok)
    out["per_speaker_recall"] = {k: float(np.mean(v)) for k, v in per.items()}

    sweep(out, model, libri, all_cases, rng)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


def sweep(out: dict, model: Path, libri: np.ndarray, all_cases, rng):
    """Recall (clean test phrases) and false wakes per hour (test-clean) at thresholds 0.3-0.99."""
    import torch

    from .model import WakeNet
    import multiprocessing as mp

    from .stream import pieces, stream_windows
    from .train import detections, score

    net = WakeNet()
    net.load_state_dict(torch.load(model.with_suffix(".pt"), map_location="cpu"))
    net.to("cuda" if torch.cuda.is_available() else "cpu")
    pos_scores = [score(net, stream_windows(a)[0]) for (_, _, (a, _, _)) in all_cases]
    neg_scores = []
    with mp.Pool(4) as pool:
        for X, _ in pool.imap(stream_windows, pieces([libri]), chunksize=1):
            neg_scores.append(score(net, X))
    hours = len(libri) / 16000 / 3600
    rows = []
    for thr in np.round(np.arange(0.3, 1.0, 0.02), 2):
        rec = float(np.mean([detections(s, thr) > 0 for s in pos_scores]))
        fa = sum(detections(s, thr) for s in neg_scores) / hours
        rows.append({"threshold": float(thr), "recall": rec, "false_per_hour": fa})
    out["sweep"] = rows
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["false_per_hour"] for r in rows], [1 - r["recall"] for r in rows], marker="o", ms=3)
        chosen = min(rows, key=lambda r: abs(r["threshold"] - out["threshold"]))
        ax.scatter([chosen["false_per_hour"]], [1 - chosen["recall"]], color="red", zorder=3, label=f"threshold {out['threshold']}")
        ax.set_xlabel("false wakes per hour (LibriSpeech test-clean)")
        ax.set_ylabel("missed wake phrases (clean, unseen voices)")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        (ROOT / "docs").mkdir(exist_ok=True)
        fig.savefig(ROOT / "docs" / "wake.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
