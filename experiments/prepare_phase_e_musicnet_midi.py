"""Prepare a compact MusicNet reference-MIDI token corpus for Phase E.

The official MusicNet Zenodo record includes a small reference-MIDI archive
alongside the much larger audio/label archive. This script mirrors the MAESTRO
MIDI preparation path and turns a deterministic MusicNet MIDI slice into an
ASCII event stream suitable for the byte-LM experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MIDI_URL = "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz?download=1"
DEFAULT_METADATA_URL = "https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1"
DEFAULT_MIDI_MD5 = "b5fa98a113bfc51c8a445def9f24dc7e"
DEFAULT_METADATA_MD5 = "1caef62cee9c875235e62aac368b49d8"


@dataclass(frozen=True)
class Piece:
    path: str
    musicnet_id: str
    composer: str
    composition: str
    movement: str
    ensemble: str
    source: str
    seconds: float


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_file(args.midi_url, args.archive, force=args.force_download, timeout=args.timeout)
        download_file(args.metadata_url, args.metadata, force=args.force_download, timeout=args.timeout)
    if not args.archive.exists():
        raise SystemExit(f"missing MIDI archive: {args.archive} (use --download or --archive)")
    if not args.metadata.exists():
        raise SystemExit(f"missing metadata CSV: {args.metadata} (use --download or --metadata)")
    if args.verify_md5:
        verify_md5(args.archive, args.midi_md5)
        verify_md5(args.metadata, args.metadata_md5)

    candidate_pieces, metadata_rows, archive_members = select_pieces(args)
    if not candidate_pieces:
        raise SystemExit("no MusicNet MIDI pieces selected")
    corpus, token_stats, pieces = tokenize_pieces(args.archive, candidate_pieces, args)
    if not pieces:
        raise SystemExit("no readable MusicNet MIDI pieces selected")

    out_path = args.out if args.out is not None else default_output_path(args, len(pieces))
    if out_path.exists() and not args.force:
        raise SystemExit(f"output already exists: {out_path} (use --force)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")

    meta = {
        "name": "musicnet_reference_midi_token_corpus",
        "source_urls": {
            "midi": args.midi_url,
            "metadata": args.metadata_url,
        },
        "archive": file_meta(args.archive),
        "metadata_file": file_meta(args.metadata),
        "output": file_meta(out_path),
        "selection": args.selection,
        "max_files": args.max_files,
        "seed": args.seed,
        "composer": args.composer,
        "ensemble": args.ensemble,
        "candidate_pieces": len(candidate_pieces),
        "grid_ms": args.grid_ms,
        "max_delta": args.max_delta,
        "include_controls": args.include_controls,
        "include_programs": args.include_programs,
        "pieces": [piece.__dict__ for piece in pieces],
        "metadata_rows": metadata_rows,
        "archive_midi_members": archive_members,
        "token_stats": token_stats,
        "schema": {
            "piece": "PIECE|musicnet_id|composer|composition|movement|seconds header followed by quantized MIDI event tokens",
            "note_on": "d{delta}.on{pitch}.v{velocity}.c{channel}",
            "note_off": "d{delta}.off{pitch}.c{channel}",
            "program": "d{delta}.pg{program}.c{channel}",
            "control": "d{delta}.cc{control}.v{value}.c{channel}",
        },
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote corpus: {out_path}")
    print(f"Wrote metadata: {meta_path}")
    print(
        json.dumps(
            {
                "selection": args.selection,
                "max_files": args.max_files,
                "grid_ms": args.grid_ms,
                "token_stats": token_stats,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/musicnet_midi"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=Path("data/musicnet_midi/musicnet_midis.tar.gz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/musicnet_midi/musicnet_metadata.csv"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--midi-url", default=DEFAULT_MIDI_URL)
    parser.add_argument("--metadata-url", default=DEFAULT_METADATA_URL)
    parser.add_argument("--midi-md5", default=DEFAULT_MIDI_MD5)
    parser.add_argument("--metadata-md5", default=DEFAULT_METADATA_MD5)
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--selection", choices=("shortest", "longest", "random", "metadata"), default="shortest")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--composer", default=None, help="Optional exact composer filter, e.g. Mozart.")
    parser.add_argument("--ensemble", default=None, help="Optional exact ensemble filter, e.g. Solo Piano.")
    parser.add_argument("--grid-ms", type=int, default=20)
    parser.add_argument("--max-delta", type=int, default=999)
    parser.add_argument("--max-events-per-piece", type=int, default=20000)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--include-programs", action="store_true")
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


def default_output_path(args: argparse.Namespace, pieces: int) -> Path:
    suffixes = ["musicnet", args.selection, str(pieces)]
    if args.composer:
        suffixes.append(clean_text(args.composer).lower())
    if args.ensemble:
        suffixes.append(clean_text(args.ensemble).lower())
    if args.include_controls:
        suffixes.append("controls")
    if args.include_programs:
        suffixes.append("programs")
    return args.out_dir / ("_".join(suffixes) + ".txt")


def download_file(url: str, path: Path, *, force: bool, timeout: float) -> None:
    if path.exists() and not force:
        print(f"Already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        path.write_bytes(response.read())
    print(f"Wrote: {path}")


def verify_md5(path: Path, expected: str) -> None:
    found = md5_file(path)
    if found != expected:
        raise SystemExit(f"MD5 mismatch for {path}: expected {expected}, found {found}")
    print(f"Verified MD5: {path}")


def select_pieces(args: argparse.Namespace) -> tuple[list[Piece], int, int]:
    rows = read_metadata(args.metadata)
    midi_paths = midi_members_by_id(args.archive)
    pieces = []
    for row in rows:
        musicnet_id = row.get("id", "")
        if musicnet_id not in midi_paths:
            continue
        composer = row.get("composer", "")
        ensemble = row.get("ensemble", "")
        if args.composer is not None and composer != args.composer:
            continue
        if args.ensemble is not None and ensemble != args.ensemble:
            continue
        pieces.append(
            Piece(
                path=midi_paths[musicnet_id],
                musicnet_id=musicnet_id,
                composer=composer,
                composition=row.get("composition", ""),
                movement=row.get("movement", ""),
                ensemble=ensemble,
                source=row.get("source", ""),
                seconds=float(row.get("seconds") or 0.0),
            )
        )

    if args.selection == "shortest":
        pieces.sort(key=lambda piece: (piece.seconds, piece.musicnet_id, piece.path))
    elif args.selection == "longest":
        pieces.sort(key=lambda piece: (-piece.seconds, piece.musicnet_id, piece.path))
    elif args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(pieces)
    else:
        pieces.sort(key=lambda piece: int(piece.musicnet_id))
    return pieces, len(rows), len(midi_paths)


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def midi_members_by_id(archive_path: Path) -> dict[str, str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        paths = {}
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith((".mid", ".midi")):
                continue
            match = re.search(r"/(\d+)[^/]*\.midi?$", member.name, flags=re.IGNORECASE)
            if match:
                paths[match.group(1)] = member.name
        return paths


def tokenize_pieces(archive_path: Path, pieces: list[Piece], args: argparse.Namespace) -> tuple[str, dict[str, int], list[Piece]]:
    try:
        import mido
    except Exception as exc:  # pragma: no cover - command-line dependency guard
        raise SystemExit("This script requires mido. Try the project venv used for PyTorch runs.") from exc

    lines = [
        "# PHASE_E_MUSICNET_REFERENCE_MIDI_TOKEN_CORPUS",
        f"# selection={args.selection} pieces={len(pieces)} grid_ms={args.grid_ms}",
        "# source=MusicNet_Zenodo_5120004_reference_midis",
        "",
    ]
    total_events = 0
    note_on = 0
    note_off = 0
    controls = 0
    programs = 0
    skipped_messages = 0
    skipped_pieces = 0
    tokenized_pieces: list[Piece] = []

    with tarfile.open(archive_path, "r:gz") as archive:
        for piece in pieces:
            if len(tokenized_pieces) >= args.max_files:
                break
            member = archive.getmember(piece.path)
            file_obj = archive.extractfile(member)
            if file_obj is None:
                skipped_pieces += 1
                continue
            try:
                midi = mido.MidiFile(file=io.BytesIO(file_obj.read()))
            except Exception:
                skipped_pieces += 1
                continue
            index = len(tokenized_pieces)
            header = (
                f"PIECE={index:04d}|ID={piece.musicnet_id}|SEC={piece.seconds:.2f}|"
                f"COMP={clean_text(piece.composer)}|ENS={clean_text(piece.ensemble)}|"
                f"WORK={clean_text(piece.composition)}|MOVE={clean_text(piece.movement)}|"
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
                elif message.type in ("note_off", "note_on") and hasattr(message, "note"):
                    delta = flush_delta(pending_delta, args.max_delta)
                    pending_delta = 0
                    channel = getattr(message, "channel", 0)
                    tokens.append(f"d{delta:03d}.off{message.note:03d}.c{channel}")
                    note_off += 1
                elif args.include_programs and message.type == "program_change":
                    delta = flush_delta(pending_delta, args.max_delta)
                    pending_delta = 0
                    tokens.append(f"d{delta:03d}.pg{message.program:03d}.c{message.channel}")
                    programs += 1
                elif args.include_controls and message.type == "control_change":
                    delta = flush_delta(pending_delta, args.max_delta)
                    pending_delta = 0
                    tokens.append(f"d{delta:03d}.cc{message.control:03d}.v{message.value:03d}.c{message.channel}")
                    controls += 1
                else:
                    skipped_messages += 1
            total_events += len(tokens)
            lines.append(header + " ".join(tokens))
            tokenized_pieces.append(piece)

    return "\n".join(lines) + "\n", {
        "pieces": len(tokenized_pieces),
        "events": total_events,
        "note_on": note_on,
        "note_off": note_off,
        "control": controls,
        "program": programs,
        "skipped_messages": skipped_messages,
        "skipped_pieces": skipped_pieces,
    }, tokenized_pieces


def quantized_delta(seconds: float, grid_ms: int) -> int:
    return max(0, int(round(float(seconds) * 1000.0 / float(grid_ms))))


def flush_delta(delta: int, max_delta: int) -> int:
    return min(max(0, delta), max_delta)


def clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ .,+:/()-]+", "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    return text[:80] if text else "NA"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, str | int]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


if __name__ == "__main__":
    main()
