"""Trains the wake word detector.

    python -m courier_wake.train

1. Windows from the synthetic speech (data/synth) and real speech (LibriSpeech dev-clean, 32 of its
   40 speakers; the other 8 are for validation), augmented, as speech embeddings from the engine.
2. A first model.
3. Hard negatives: the first model streams over the training speech, and every window it scores
   above 0.2 that is not the wake phrase becomes a training example.
4. The final model, trained on both.
5. The threshold: the highest recall on the validation speakers (LibriTTS-R 850-903) that keeps false
   wakes on the validation speech under one per hour.
The model goes to models/wake/hey_courier.onnx, its threshold and validation numbers to
models/wake/hey_courier.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import dataset as D
from .engine import EMBEDDINGS
from .stream import pieces, stream_windows
from .model import WakeNet, export
from .synth import SYNTH, load

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "wake"
FEAT = D.DATA / "features"
REFRACTORY_HOPS = int(1.5 / 0.08)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data -----------------------------------------------------------------------------------------


def build_split(split: str, rng, workers: int):
    """Feature windows and labels of one split ("train" or "val")."""
    pos, _ = load(SYNTH / f"{split}_pos.npz")
    neg, _ = load(SYNTH / f"{split}_neg.npz")
    conf, _ = load(SYNTH / f"{split}_conf.npz")

    speakers = D.libri_speakers("dev-clean")
    real = D.libri_audio(speakers[:32] if split == "train" else speakers[32:])
    others = neg + [(r * 32767).astype(np.int16) for r in real[:300]]
    bank = D.noise_bank(real, rng)
    scale = 1 if split == "train" else 0.25
    plan = [
        ("pos", pos, int(3 * len(pos) * scale), 1),
        ("conf", conf, int(2 * len(conf) * scale), 0),
        ("partial", pos, int(len(pos) * scale), 0),
        ("tail", pos, int(0.6 * len(pos) * scale), 0),
        ("speech", neg, int(2 * len(neg) * scale), 0),
        ("speech", real, int(28000 * scale), 0),
        ("noise", pos, int(3000 * scale), 0),
    ]
    X, y = [], []
    for kind, clips, n, label in plan:
        w = D.make_windows(kind, clips, others, n, rng)
        X.append(D.features(w, bank, seed=int(rng.integers(1 << 30)), workers=workers))
        y.append(np.full(len(w), label, np.float32))
        print(f"  {split} {kind}: {len(w)} windows", flush=True)
    return np.concatenate(X), np.concatenate(y), bank


@torch.no_grad()
def score(model: nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    model.to(device())
    dev = device()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32)).to(dev)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def detections(scores: np.ndarray, threshold: float, hits: int = 2) -> int:
    """Counts detections the way the engine does: `hits` scores in a row, then 1.5 s of quiet."""
    n, run, quiet_until = 0, 0, -1
    for i, s in enumerate(scores):
        if i < quiet_until:
            run = 0
            continue
        run = run + 1 if s >= threshold else 0
        if run >= hits:
            n += 1
            run = 0
            quiet_until = i + REFRACTORY_HOPS + 1
    return n


# --- training -------------------------------------------------------------------------------------


def spec_augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Drops a tenth of the embedding values and adds a little noise: the features vary between
    microphones and rooms more than the augmented training audio covers."""
    keep = (torch.rand(x.shape, generator=rng) > 0.1).to(x.device)
    noise = torch.randn(x.shape, generator=rng).to(x.device) * 0.05 * x.std()
    return x * keep + noise


def fit(model, X, y, Xv, yv, epochs: int, lr: float, seed: int, log: list):
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * ((len(X) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X)  # float16 on the CPU; batches go to the GPU as float32
    yt = torch.from_numpy(y)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        total, t = 0.0, time.time()
        for i in range(0, len(X), 256):
            idx = perm[i : i + 256]
            xb = Xt[idx].to(dev).float()
            yb = yt[idx].to(dev)
            xb = spec_augment(xb, g)
            # Label smoothing keeps scores from saturating, so the threshold has room to work.
            loss = loss_fn(model(xb), yb * 0.95 + 0.025)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        s = score(model, Xv)
        pos, neg = s[yv == 1], s[yv == 0]
        thr = np.quantile(neg, 0.999)
        rec = float((pos > thr).mean())
        line = f"epoch {ep + 1}/{epochs}: loss {total / len(X):.4f}, val recall at 0.1% false windows {rec:.3f} ({time.time() - t:.0f} s)"
        print(line, flush=True)
        log.append(line)
    return model


def mine(model, sources: list[np.ndarray], bank, rng, workers: int, above: float = 0.2, cap: int = 20000) -> np.ndarray:
    """Windows of non-wake speech the model scores above `above`, as augmented features (two each)."""
    import multiprocessing as mp

    found = []
    parts = pieces(sources)
    # The embeddings are the slow part (a CPU core does about 25 times real time): one process each.
    with mp.Pool(min(workers, 4)) as pool:
        for a, (X, starts) in zip(parts, pool.imap(stream_windows, parts, chunksize=1)):
            if len(X) == 0:
                continue
            s = score(model, X)
            for i in np.where(s > above)[0]:
                found.append(a[starts[i] : starts[i] + D.W])
    print(f"  mined {len(found)} hard negative windows", flush=True)
    if len(found) > cap:
        found = [found[i] for i in rng.choice(len(found), cap, replace=False)]
    if not found:
        return np.zeros((0, EMBEDDINGS, 96), np.float16)
    # A few thousand windows: computed here (a second pool right after the first has hung on Windows).
    return D.features(found + found, bank, seed=int(rng.integers(1 << 30)), workers=1)


def streams(clips: list[np.ndarray], rng, gap_s=(0.3, 1.5)) -> np.ndarray:
    """Clips joined with pauses into one long recording."""
    parts = []
    for c in clips:
        parts.append(D.f32(c))
        parts.append(np.zeros(int(rng.uniform(*gap_s) * 16000), np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def choose_threshold(model, rng) -> dict:
    """Recall on validation phrases in context vs. false wakes per hour on validation speech."""
    pos, _ = load(SYNTH / "val_pos.npz")
    neg, _ = load(SYNTH / "val_neg.npz")
    conf, _ = load(SYNTH / "val_conf.npz")
    real = D.libri_audio(D.libri_speakers("dev-clean")[32:])
    # Positives: each phrase in its own short recording, so each can be found or missed.
    pos_scores = []
    for c in pos:
        # 2.5 s before the phrase: the detector scores once it has 2 s of audio, as it always has
        # when listening (a recording that starts just before the phrase would be missed).
        a = np.concatenate([np.zeros(40000, np.float32), D.f32(c), D.f32(neg[rng.integers(len(neg))])[:12000], np.zeros(8000, np.float32)])
        X, _ = stream_windows(a)
        pos_scores.append(score(model, X))
    negatives = [streams(neg, rng), streams(conf, rng), np.concatenate(real)]
    neg_scores = [score(model, stream_windows(s)[0]) for s in negatives]
    hours = sum(len(s) for s in negatives) / 16000 / 3600
    best = None
    for thr in np.round(np.arange(0.30, 0.99, 0.01), 2):
        rec = float(np.mean([detections(s, thr) > 0 for s in pos_scores]))
        fa = sum(detections(s, thr) for s in neg_scores) / hours
        if fa <= 1.0 and (best is None or rec > best["recall"]):
            best = {"threshold": float(thr), "recall": rec, "false_per_hour": fa}
    best["validation_hours"] = hours
    best["validation_phrases"] = len(pos_scores)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold-only", action="store_true", help="choose the threshold again for the trained model")
    args = ap.parse_args()
    if args.threshold_only:
        model = WakeNet()
        model.load_state_dict(torch.load(OUT / "hey_courier.pt", map_location="cpu"))
        model.to(device())
        chosen = choose_threshold(model, np.random.default_rng(args.seed))
        info = json.loads((OUT / "hey_courier.json").read_text())
        info.update({"threshold": chosen["threshold"], "validation": chosen})
        (OUT / "hey_courier.json").write_text(json.dumps(info, indent=2))
        print(f"threshold {chosen}", flush=True)
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    FEAT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cache = FEAT / f"windows-{args.seed}.npz"
    if cache.exists():
        z = np.load(cache)
        X, y, Xv, yv = z["X"], z["y"], z["Xv"], z["yv"]
        bank = D.noise_bank(D.libri_audio(D.libri_speakers("dev-clean")[:32], limit_s=1200), rng)
    else:
        X, y, bank = build_split("train", rng, args.workers)
        Xv, yv, _ = build_split("val", rng, args.workers)
        np.savez(cache, X=X, y=y, Xv=Xv, yv=yv)
    print(f"windows: {len(X)} train ({int(y.sum())} positive), {len(Xv)} validation; {time.time() - t0:.0f} s", flush=True)

    log = []
    hard_cache = FEAT / f"hard-{args.seed}.npz"
    if hard_cache.exists():
        hard = np.load(hard_cache)["hard"]
    else:
        model = fit(WakeNet(), X, y, Xv, yv, args.epochs, 3e-3, args.seed, log)
        # Hard negatives from the training speech, synthetic and real.
        neg, _ = load(SYNTH / "train_neg.npz")
        conf, _ = load(SYNTH / "train_conf.npz")
        real = D.libri_audio(D.libri_speakers("dev-clean")[:32])
        hard = mine(model, [streams(neg, rng), streams(conf, rng)] + real, bank, rng, args.workers)
        np.savez(hard_cache, hard=hard)
    X2 = np.concatenate([X, hard])
    y2 = np.concatenate([y, np.zeros(len(hard), np.float32)])
    torch.manual_seed(args.seed + 1)
    model = fit(WakeNet(), X2, y2, Xv, yv, args.epochs, 3e-3, args.seed + 1, log)

    chosen = choose_threshold(model, rng)
    print(f"threshold {chosen}", flush=True)
    torch.save(model.state_dict(), OUT / "hey_courier.pt")
    export(model, OUT / "hey_courier.onnx")
    info = {
        "phrase": "Hey Courier",
        "threshold": chosen["threshold"],
        "hits": 2,
        "refractory_ms": 1500,
        "validation": chosen,
        "train_windows": int(len(X2)),
        "hard_negatives": int(len(hard)),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "log": log,
    }
    (OUT / "hey_courier.json").write_text(json.dumps(info, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
