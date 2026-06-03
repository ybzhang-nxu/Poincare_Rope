"""Download lightweight text corpora for Phase D byte-level LM experiments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path


CORPORA = {
    "tiny_shakespeare": {
        "kind": "text",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "path": Path("data/tiny_shakespeare/input.txt"),
        "min_bytes": 500_000,
    },
    "wikitext2": {
        "kind": "text",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "path": Path("data/wikitext2/train.txt"),
        "min_bytes": 5_000_000,
    },
    "gutenberg_warpeace": {
        "kind": "text",
        "url": "https://www.gutenberg.org/files/2600/2600-0.txt",
        "path": Path("data/gutenberg_warpeace/war_and_peace.txt"),
        "min_bytes": 2_000_000,
    },
}


def main() -> None:
    args = parse_args()
    spec = CORPORA[args.name]
    out_path = args.out if args.out is not None else spec["path"]
    if spec["kind"] == "zip" and args.out is not None:
        raise SystemExit("--out is only supported for direct text corpora")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        raw = out_path.read_bytes()
        print(f"Already exists: {out_path}")
    elif spec["kind"] == "zip":
        raw = download_zip_corpus(args.name, spec, timeout=args.timeout)
    else:
        url = str(spec["url"])
        print(f"Downloading {args.name} from {url}")
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            raw = response.read()
        out_path.write_bytes(raw)
        print(f"Wrote: {out_path}")

    if len(raw) < int(spec["min_bytes"]):
        raise SystemExit(f"corpus is unexpectedly small: {len(raw)} bytes")

    meta = {
        "name": args.name,
        "url": spec["url"],
        "path": str(out_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")
    print(json.dumps(meta, indent=2))


def download_zip_corpus(name: str, spec: dict, *, timeout: float) -> bytes:
    url = str(spec["url"])
    print(f"Downloading {name} from {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        archive_raw = response.read()

    selected_raw = None
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        for member, path in spec["members"].items():
            raw = archive.read(member)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            print(f"Wrote: {path}")
            if path == spec["path"]:
                selected_raw = raw
    if selected_raw is None:
        raise SystemExit(f"selected path was not written for {name}: {spec['path']}")
    archive_meta = {
        "name": name,
        "url": spec["url"],
        "archive_bytes": len(archive_raw),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "members": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in spec["members"].values()
        },
    }
    archive_meta_path = spec["path"].parent / "archive.json"
    archive_meta_path.write_text(json.dumps(archive_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote archive metadata: {archive_meta_path}")
    return selected_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted(CORPORA), default="tiny_shakespeare")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
