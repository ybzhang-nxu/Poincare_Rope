"""Generate a deterministic symbolic music-like corpus for Phase E.

This is a lightweight scaffold, not a replacement for MAESTRO or MusicNet.
It creates compact ASCII MIDI-ish event streams with motif returns, phrase
timing, rhythm envelopes, and a matched bar-shuffle control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


ROOTS = ("C", "D", "E", "F", "G", "A", "B")
MODE_INTERVALS = {
    "ion": (0, 2, 4, 5, 7, 9, 11),
    "dor": (0, 2, 3, 5, 7, 9, 10),
    "min": (0, 2, 3, 5, 7, 8, 10),
}
FORM = ("A", "A", "B", "A", "C", "B", "A", "D")
KEY_CYCLE = (0, 5, 7, 2, 9, 4, 11, 6)
RHYTHM_ENVELOPES = ("dense", "sync", "pedal", "answer")


@dataclass(frozen=True)
class Event:
    step: int
    degree: int
    octave: int
    duration: int
    velocity: int
    channel: int


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motifs = build_motif_bank(rng)
    bars, return_counts = generate_bars(args, motifs, rng)
    variation_bars, variation_return_counts = generate_bars(
        args,
        motifs,
        random.Random(args.seed + 2027),
        return_mode="variation",
    )
    motif_rich = render_corpus(
        bars,
        name="motif_rich",
        seed=args.seed,
        bars_per_phrase=args.bars_per_phrase,
        return_counts=return_counts,
    )

    shuffled = list(bars)
    random.Random(args.seed + 1009).shuffle(shuffled)
    bar_shuffle_control = render_corpus(
        shuffled,
        name="bar_shuffle_control",
        seed=args.seed,
        bars_per_phrase=args.bars_per_phrase,
        return_counts=return_counts,
    )
    motif_variation = render_corpus(
        variation_bars,
        name="motif_variation",
        seed=args.seed,
        bars_per_phrase=args.bars_per_phrase,
        return_counts=variation_return_counts,
    )
    variation_shuffled = list(variation_bars)
    random.Random(args.seed + 3037).shuffle(variation_shuffled)
    variation_shuffle_control = render_corpus(
        variation_shuffled,
        name="variation_shuffle_control",
        seed=args.seed,
        bars_per_phrase=args.bars_per_phrase,
        return_counts=variation_return_counts,
    )

    outputs = {
        "motif_rich": args.out_dir / "motif_rich.txt",
        "bar_shuffle_control": args.out_dir / "bar_shuffle_control.txt",
        "motif_variation": args.out_dir / "motif_variation.txt",
        "variation_shuffle_control": args.out_dir / "variation_shuffle_control.txt",
    }
    write_text(outputs["motif_rich"], motif_rich, force=args.force)
    write_text(outputs["bar_shuffle_control"], bar_shuffle_control, force=args.force)
    write_text(outputs["motif_variation"], motif_variation, force=args.force)
    write_text(outputs["variation_shuffle_control"], variation_shuffle_control, force=args.force)

    meta = {
        "name": "phase_e_symbolic_music_like",
        "description": "ASCII MIDI-ish corpus with motif returns and matched bar-shuffle control.",
        "seed": args.seed,
        "bars": args.bars,
        "bars_per_phrase": args.bars_per_phrase,
        "bar_steps": 16,
        "return_lags_bars": sorted(return_counts),
        "return_counts": {str(key): value for key, value in sorted(return_counts.items())},
        "variation_return_counts": {str(key): value for key, value in sorted(variation_return_counts.items())},
        "outputs": {name: file_meta(path) for name, path in outputs.items()},
        "schema": {
            "bar": "SEC/PH/M/K/R/H header followed by MIDI-ish step:pitch:velocity:duration:channel events",
            "event": "t{step}.n{pitch}.v{velocity}.d{duration}.c{channel}",
        },
    }
    meta_path = args.out_dir / "metadata.json"
    if meta_path.exists() and not args.force:
        raise SystemExit(f"metadata already exists: {meta_path} (use --force)")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {outputs['motif_rich']}")
    print(f"Wrote: {outputs['bar_shuffle_control']}")
    print(f"Wrote: {outputs['motif_variation']}")
    print(f"Wrote: {outputs['variation_shuffle_control']}")
    print(f"Wrote metadata: {meta_path}")
    print(json.dumps(meta, indent=2))


def build_motif_bank(rng: random.Random) -> dict[str, list[Event]]:
    bank = {}
    for section, contour in {
        "A": (0, 2, 4, 5, 4, 2, 1, 0, 2, 4, 7, 5, 4, 2, 0, -1),
        "B": (4, 5, 7, 9, 7, 5, 4, 2, 0, 2, 4, 5, 7, 5, 2, 0),
        "C": (0, 0, 7, 5, 0, 0, 4, 2, -1, 0, 2, 4, 5, 4, 2, 0),
        "D": (7, 5, 4, 2, 4, 5, 7, 9, 11, 9, 7, 5, 4, 2, 0, 2),
    }.items():
        for variant in range(4):
            events = []
            for step, degree in enumerate(contour):
                if rng.random() < 0.08 and step not in (0, 8):
                    continue
                sync = 1 if section in ("B", "D") and step % 4 == 3 else 0
                duration = 1 + int((step + variant + sync) % 4 == 0)
                velocity = 54 + 5 * ((step + variant) % 4) + rng.randint(0, 7)
                octave = 4 + int(degree >= 7) - int(degree < 0)
                events.append(
                    Event(
                        step=step,
                        degree=degree + variant - 1,
                        octave=octave,
                        duration=duration,
                        velocity=velocity,
                        channel=variant % 2,
                    )
                )
            bank[f"{section}{variant}"] = events
    return bank


def generate_bars(
    args: argparse.Namespace,
    motifs: dict[str, list[Event]],
    rng: random.Random,
    *,
    return_mode: str = "exact",
) -> tuple[list[str], dict[int, int]]:
    if return_mode not in ("exact", "variation"):
        raise ValueError(f"unknown return_mode: {return_mode}")
    bars = []
    return_counts = {8: 0, 16: 0, 32: 0, 64: 0}
    for bar_index in range(args.bars):
        lag = selected_return_lag(bar_index)
        if lag is not None and bar_index >= lag:
            source_bar = bars[bar_index - lag]
            if return_mode == "variation":
                source_bar = vary_return_bar(source_bar, lag=lag, bar_index=bar_index, rng=rng)
            bars.append(source_bar)
            return_counts[lag] += 1
            continue
        bars.append(render_new_bar(bar_index, args.bars_per_phrase, motifs, rng))
    return bars, return_counts


def selected_return_lag(bar_index: int) -> int | None:
    if bar_index % 64 in range(0, 8):
        return 64
    if bar_index % 32 in range(16, 20):
        return 32
    if bar_index % 16 in range(8, 10):
        return 16
    if bar_index % 8 == 7:
        return 8
    return None


def render_new_bar(
    bar_index: int,
    bars_per_phrase: int,
    motifs: dict[str, list[Event]],
    rng: random.Random,
) -> str:
    phrase_index = bar_index // bars_per_phrase
    phrase_pos = bar_index % bars_per_phrase
    section = FORM[phrase_index % len(FORM)]
    variant = (phrase_index + phrase_pos) % 4
    motif_name = f"{section}{variant}"
    mode = ("ion", "dor", "min")[(phrase_index // 4 + variant) % 3]
    root_offset = KEY_CYCLE[(phrase_index // 2 + phrase_pos) % len(KEY_CYCLE)]
    root = ROOTS[root_offset % len(ROOTS)]
    rhythm = RHYTHM_ENVELOPES[(phrase_pos + phrase_index) % len(RHYTHM_ENVELOPES)]
    humanize = (phrase_index * 3 + phrase_pos * 5) % 11
    header = (
        f"SEC={section}|PH={phrase_pos:02d}|M={motif_name}|K={root}{root_offset:+03d}|"
        f"MODE={mode}|R={rhythm}|H={humanize:02d}|"
    )
    events = [
        render_event(event, root_offset=root_offset, mode=mode, phrase_pos=phrase_pos, rng=rng)
        for event in motifs[motif_name]
    ]
    bass = render_bass(root_offset=root_offset, phrase_pos=phrase_pos, mode=mode)
    return header + " ".join(bass + events) + "\n"


def vary_return_bar(bar: str, *, lag: int, bar_index: int, rng: random.Random) -> str:
    text = bar.rstrip("\n")
    if "|" not in text:
        return bar
    header, body = text.rsplit("|", 1)
    phrase = (bar_index // max(1, lag)) % 7
    transpose = (-2, 1, 2, -1, 3, -3, 4)[phrase]
    velocity_shift = (-3, 2, 4, -2, 1, -4, 3)[(phrase + lag) % 7]
    tokens = [
        vary_event_token(token, transpose=transpose, velocity_shift=velocity_shift, rng=rng)
        for token in body.split()
    ]
    return header + "|" + " ".join(tokens) + "\n"


def vary_event_token(
    token: str,
    *,
    transpose: int,
    velocity_shift: int,
    rng: random.Random,
) -> str:
    if not token.startswith("t") or ".n" not in token:
        return token
    parts = []
    channel = token.rsplit(".c", 1)[-1] if ".c" in token else ""
    for part in token.split("."):
        if part.startswith("n") and part[1:].isdigit():
            pitch = max(21, min(108, int(part[1:]) + transpose))
            parts.append(f"n{pitch:03d}")
        elif part.startswith("v") and part[1:].isdigit():
            velocity = max(1, min(127, int(part[1:]) + velocity_shift + rng.randint(-2, 2)))
            parts.append(f"v{velocity:03d}")
        elif part.startswith("d") and part[1:].isdigit() and channel != "9":
            duration = int(part[1:])
            if rng.random() < 0.10:
                duration = max(1, min(8, duration + rng.choice((-1, 1))))
            parts.append(f"d{duration:02d}")
        else:
            parts.append(part)
    return ".".join(parts)


def render_event(
    event: Event,
    *,
    root_offset: int,
    mode: str,
    phrase_pos: int,
    rng: random.Random,
) -> str:
    intervals = MODE_INTERVALS[mode]
    degree = event.degree
    octave_shift, scale_index = divmod(degree, len(intervals))
    pitch = 12 * (event.octave + octave_shift) + intervals[scale_index] + root_offset
    pitch = max(21, min(108, pitch))
    velocity = max(1, min(127, event.velocity + 2 * (phrase_pos % 3) + rng.randint(-1, 1)))
    return f"t{event.step:02d}.n{pitch:03d}.v{velocity:03d}.d{event.duration:02d}.c{event.channel}"


def render_bass(*, root_offset: int, phrase_pos: int, mode: str) -> list[str]:
    intervals = MODE_INTERVALS[mode]
    root_pitch = 36 + root_offset + intervals[(phrase_pos * 2) % len(intervals)]
    fifth_pitch = min(72, root_pitch + 7)
    return [
        f"t00.n{root_pitch:03d}.v082.d04.c9",
        f"t08.n{fifth_pitch:03d}.v074.d04.c9",
    ]


def render_corpus(
    bars: list[str],
    *,
    name: str,
    seed: int,
    bars_per_phrase: int,
    return_counts: dict[int, int],
) -> str:
    header = [
        f"# PHASE_E_SYMBOLIC_MUSIC name={name}",
        f"# seed={seed} bars={len(bars)} bars_per_phrase={bars_per_phrase}",
        "# structure=motif_return rhythm_envelope phrase_timing long_range_repetition",
        "# return_counts="
        + ",".join(f"{lag}:{count}" for lag, count in sorted(return_counts.items())),
        "",
    ]
    return "\n".join(header) + "".join(bars)


def write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"output already exists: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_meta(path: Path) -> dict[str, str | int]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase_e_music"))
    parser.add_argument("--bars", type=int, default=8192)
    parser.add_argument("--bars-per-phrase", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bars < 128:
        raise SystemExit("--bars must be at least 128")
    if args.bars_per_phrase < 2:
        raise SystemExit("--bars-per-phrase must be at least 2")
    return args


if __name__ == "__main__":
    main()
