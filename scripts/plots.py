"""Plots for the README from results/*.json (run with the wakeword venv, which has matplotlib):

    wakeword/.venv/Scripts/python scripts/plots.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def latency():
    runs = [("speculative start (Courier)", RESULTS / "latency.json"), ("model starts at the end of speech", RESULTS / "latency-no-speculation.json")]
    runs = [(name, json.loads(p.read_text())) for name, p in runs if p.exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), sharey=True, squeeze=False)
    for ax, (name, d) in zip(axes[0], runs):
        rows = [r for r in d["requests"] if r["first_audio_after_speech_ms"] is not None]
        labels = [r["text"].rstrip(".?!")[:34] for r in rows]
        end = [max(0, r["endpoint_after_speech_ms"] or 0) for r in rows]
        rest = [r["first_audio_after_speech_ms"] - e for r, e in zip(rows, end)]
        y = range(len(rows))
        ax.barh(y, end, color="#8fa8c8", label="until the end of speech is confirmed")
        ax.barh(y, rest, left=end, color="#2f5d8a", label="then until the first word is heard")
        ax.axvline(1000, color="#c0392b", lw=1, ls="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("ms after you stop speaking")
        med = d["summary"]["first_audio_ms_median"]
        ax.set_title(f"{name}\nmedian {med:.0f} ms", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "latency.png", dpi=150)
    print("wrote docs/latency.png")


def endpoint():
    p = RESULTS / "endpoint.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())["results"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in d:
        x, yv = r["delay_ms_median"], r["cut_early_rate"] * 100
        courier = r["config"].startswith("transcript")
        ax.scatter([x], [yv], s=60, color="#c0392b" if courier else "#2f5d8a", zorder=3)
        ax.annotate(r["config"], (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("median wait after the end of speech (ms)")
    ax.set_ylabel("requests cut off mid-sentence (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "endpoint.png", dpi=150)
    print("wrote docs/endpoint.png")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    latency()
    endpoint()
