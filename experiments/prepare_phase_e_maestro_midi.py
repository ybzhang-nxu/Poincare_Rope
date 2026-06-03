"""Prepare a compact MAESTRO MIDI token corpus for Phase E.

The script can download the official MAESTRO v3.0.0 MIDI-only archive, tokenize
selected MIDI files into an ASCII event stream, and write a byte-LM compatible
text corpus. It is intentionally lightweight and uses `mido` instead of a large
music preprocessing stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
DEFAULT_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"


@dataclass(frozen=True)
class Piece:
    path: str
    split: str
    composer: str
    title: str
    duration: float


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = args.archive
    if args.download:
        download_archive(args.url, archive_path, force=args.force_download, timeout=args.timeout)
    if not archive_path.exists():
        raise SystemExit(f"missing archive: {archive_path} (use --download or --archive)")
    if args.verify_sha256:
        verify_sha256(archive_path, args.sha256)

    pieces, metadata = select_pieces(archive_path, args)
    if not pieces:
        raise SystemExit("no MIDI pieces selected")
    corpus, token_stats = tokenize_pieces(archive_path, pieces, args)

    out_path = args.out if args.out is not None else args.out_dir / f"maestro_v3_{args.split}_{len(pieces)}.txt"
    if out_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {out_path} (use --force)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")

    meta = {
        "name": "maestro_v3_midi_token_corpus",
        "source_url": args.url,
        "archive": file_meta(archive_path),
        "output": file_meta(out_path),
        "split": args.split,
        "selection": args.selection,
        "max_files": args.max_files,
        "grid_ms": args.grid_ms,
        "max_delta": args.max_delta,
        "include_controls": args.include_controls,
        "pieces": [piece.__dict__ for piece in pieces],
        "metadata_rows": metadata,
        "token_stats": token_stats,
        "schema": {
            "piece": "PIECE|split|composer|title|duration header followed by quantized MIDI event tokens",
            "note_on": "d{delta}.on{pitch}.v{velocity}.c{channel}",
            "note_off": "d{delta}.off{pitch}.c{channel}",
            "control": "d{delta}.cc{control}.v{value}.c{channel}",
        },
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote corpus: {out_path}")
    print(f"Wrote metadata: {meta_path}")
    print(json.dumps({key: meta[key] for key in ("split", "selection", "max_files", "grid_ms", "token_stats")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/maestro_midi"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=Path("data/maestro_midi/maestro-v3.0.0-midi.zip"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--selection", choices=("shortest", "longest", "random", "metadata"), default="shortest")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-ms", type=int, default=20)
    parser.add_argument("--max-delta", type=int, default=999)
    parser.add_argument("--max-events-per-piece", type=int, default=20000)
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    if args.max_files < 1:
        raise SystemExit("--max-files must be positive")
    if args.grid_ms < 1:
        raise SystemExit("--grid-ms must be positive")
    if args.max_delta < 1:
        raise SystemExit("--max-delta must be positive")
    if args.max_events_per_piece < 1:
        raise SystemExit("--max-events-per-piece must be positive")
    return args


def download_archive(url: str, path: Path, *, force: bool, timeout: float) -> None:
    if path.exists() and not force:
        print(f"Already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MAESTRO MIDI archive from {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        path.write_bytes(response.read())
    print(f"Wrote: {path}")


def verify_sha256(path: Path, expected: str) -> None:
    found = sha256_file(path)
    if found != expected:
        raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, found {found}")
    print(f"Verified SHA256: {path}")


def select_pieces(archive_path: Path, args: argparse.Namespace) -> tuple[list[Piece], int]:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        metadata_name = next((name for name in members if name.endswith("maestro-v3.0.0.csv")), None)
        if metadata_name is None:
            raise SystemExit("metadata CSV not found in MAESTRO archive")
        raw_csv = archive.read(metadata_name).decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(raw_csv)))
        pieces = []
        for row in rows:
            split = row.get("split", "")
            if args.split != "all" and split != args.split:
                continue
            midi_path = row.get("midi_filename", "")
            if midi_path not in members:
                prefixed = f"maestro-v3.0.0/{midi_path}"
                if prefixed in members:
                    midi_path = prefixed
                else:
                    continue
            pieces.append(
                Piece(
                    path=midi_path,
                    split=split,
                    composer=row.get("canonical_composer", ""),
                    title=row.get("canonical_title", ""),
                    duration=float(row.get("duration") or 0.0),
                )
            )

    if args.selection == "shortest":
        pieces.sort(key=lambda piece: (piece.duration, piece.path))
    elif args.selection == "longest":
        pieces.sort(key=lambda piece: (-piece.duration, piece.path))
    elif args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(pieces)
    else:
        pieces.sort(key=lambda piece: piece.path)
    return pieces[: args.max_files], len(rows)


def tokenize_pieces(archive_path: Path, pieces: list[Piece], args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    try:
        import mido
    except Exception as exc:  # pragma: no cover - command-line dependency guard
        raise SystemExit("This script requires mido. Try the project venv used for PyTorch runs.") from exc

    lines = [
        "# PHASE_E_MAESTRO_MIDI_TOKEN_CORPUS",
        f"# split={args.split} selection={args.selection} pieces={len(pieces)} grid_ms={args.grid_ms}",
        "# source=MAESTRO_v3.0.0_midi_only",
        "",
    ]
    total_events = 0
    note_on = 0
    note_off = 0
    controls = 0
    skipped_messages = 0

    with zipfile.ZipFile(archive_path) as archive:
        for index, piece in enumerate(pieces):
            raw = archive.read(piece.path)
            midi = mido.MidiFile(file=io.BytesIO(raw))
            header = (
                f"PIECE={index:04d}|SPLIT={piece.split}|DUR={piece.duration:.2f}|"
                f"COMP={clean_text(piece.composer)}|TITLE={clean_text(piece.title)}|"
            )
            tokens = []
            pending_delta = 0
            for message in midi:
                pending_delta += quantized_delta(message.time, args.grid_ms)
                if len(tokens) >= args.max_events_per_piece:
                    break
                if message.type == "note_on" and message.velocity > 0:
                    delta = flush_delta(pending_delta, args.max_delta)
                    pending_delta = 0
                    tokens.append(f"d{delta:03d}.on{message.note:03d}.v{message.velocity:03d}.c{message.channel}")
                    note_on += 1
                elif message.type in ("note_off", "note_on"):
                    if hasattr(message, "note"):
                        delta = flush_delta(pending_delta, args.max_delta)
                        pending_delta = 0
                        channel = getattr(message, "channel", 0)
                        tokens.append(f"d{delta:03d}.off{message.note:03d}.c{channel}")
                        note_off += 1
                    else:
                        skipped_messages += 1
                elif args.include_controls and message.type == "control_change":
                    delta = flush_delta(pending_delta, args.max_delta)
                    pending_delta = 0
                    tokens.append(f"d{delta:03d}.cc{message.control:03d}.v{message.value:03d}.c{message.channel}")
                    controls += 1
                else:
                    skipped_messages += 1
            total_events += len(tokens)
            lines.append(header + " ".join(tokens))

    return "\n".join(lines) + "\n", {
        "pieces": len(pieces),
        "events": total_events,
        "note_on": note_on,
        "note_off": note_off,
        "control": controls,
        "skipped_messages": skipped_messages,
    }


def quantized_delta(seconds: float, grid_ms: int) -> int:
    return max(0, int(round(float(seconds) * 1000.0 / float(grid_ms))))


def flush_delta(delta: int, max_delta: int) -> int:
    return min(max(0, delta), max_delta)


def clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ .,+:/()-]+", "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    return text[:80] if text else "NA"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, str | int]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


if __name__ == "__main__":
    main()
