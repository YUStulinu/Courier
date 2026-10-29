"""Records your voice for personalizing the wake word: the phrase a number of times, and some ordinary
sentences (so the detector also learns what your voice sounds like when it is not being called).

    python -m courier_wake.record                 # into data/me/
    python -m courier_wake.personalize --voice data/me
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import rms_db, trim

DATA = Path(__file__).resolve().parents[1] / "data"

SENTENCES = [
    "What's the weather going to be like tomorrow?",
    "I need to buy milk, eggs and bread.",
    "The meeting has been moved to Thursday afternoon.",
    "Could you send me the report by Friday?",
    "My courier delivered the parcel this morning.",
    "Hey, can you hear me?",
    "Let's have dinner at eight tonight.",
    "I'll call you back in ten minutes.",
    "Turn left at the end of the street.",
    "Okay, that sounds good to me.",
    "The train was twenty minutes late again.",
    "Hey Corey, how are you doing?",
]


def take(seconds: float) -> np.ndarray:
    import sounddevice as sd

    a = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    return a[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "me"))
    ap.add_argument("--phrases", type=int, default=12)
    ap.add_argument("--sentences", type=int, default=len(SENTENCES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Say \"Hey Courier\" each time you see GO, as you would to the assistant: from where you usually sit,")
    print("some louder, some quieter, some faster. Then read a few sentences.\n")
    for i in range(args.phrases):
        input(f"[{i + 1}/{args.phrases}] Enter, then say it at GO... ")
        time.sleep(0.3)
        print("GO")
        a = trim(take(2.2))
        if len(a) < 4000 or rms_db(a) < -55:
            print("  too quiet or too short, skipped")
            continue
        sf.write(out / f"wake_{i:02d}.wav", a, 16000)
        print(f"  {len(a) / 16000:.2f} s, level {rms_db(a):.0f} dB")
    for i, s in enumerate(SENTENCES[: args.sentences]):
        input(f"[{i + 1}/{args.sentences}] Enter, then read: \"{s}\" ")
        a = trim(take(4.5))
        if len(a) < 4000:
            print("  too short, skipped")
            continue
        sf.write(out / f"other_{i:02d}.wav", a, 16000)
    print(f"\nSaved in {out}. Next: python -m courier_wake.personalize --voice {out}")


if __name__ == "__main__":
    main()
