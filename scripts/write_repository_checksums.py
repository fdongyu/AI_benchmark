#!/usr/bin/env python3
"""Write or verify SHA-256 checksums for the stable repository surface."""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("AGENTS.md"),
    Path("ARTIFACTS.sha256"),
    Path("CLAUDE.md"),
    Path("README.md"),
    Path("TRAINING_DATA.md"),
    Path("publication_manifest.json"),
    Path("analysis"),
    Path("data"),
    Path("docs"),
    Path("environment"),
    Path("model"),
    Path("provenance"),
    Path("scripts"),
    Path("src"),
    Path("tests"),
    Path("workflows"),
)
EXCLUDED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", "cache", "logs", "locks", "receipts"}
)
EXCLUDED_NAMES = frozenset({"MANIFEST.sha256"})
GENERATED_PREFIXES = (Path("analysis/figures"),)


def generated(relative: Path) -> bool:
    return any(relative == prefix or prefix in relative.parents for prefix in GENERATED_PREFIXES)


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and not generated(relative)
        and not EXCLUDED_PARTS.intersection(relative.parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix != ".pyc"
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collect_paths() -> List[Path]:
    paths = []
    for relative in INCLUDE:
        candidate = ROOT / relative
        if candidate.is_file() and included(candidate):
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if included(path))
    return sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)))


def render(paths: List[Path]) -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT)}\n" for path in paths
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare MANIFEST.sha256 with the current tree without writing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = collect_paths()
    destination = ROOT / "MANIFEST.sha256"
    expected = render(paths)

    if args.check:
        if not destination.is_file():
            print(f"{destination}: missing", file=sys.stderr)
            return 1
        current = destination.read_text(encoding="utf-8")
        if current == expected:
            print(f"{destination}: current ({len(paths)} files)")
            return 0
        current_lines = set(current.splitlines())
        expected_lines = set(expected.splitlines())
        print(
            f"{destination}: stale; "
            f"{len(expected_lines - current_lines)} added/changed and "
            f"{len(current_lines - expected_lines)} removed/changed lines",
            file=sys.stderr,
        )
        return 1

    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{int(time.time() * 1_000_000_000)}"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(expected)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(f"{destination}: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
