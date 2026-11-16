#!/usr/bin/env python3
"""Run lightweight, dependency-free checks before publishing the repository."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
JUNK_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cache",
    "logs",
    "locks",
    "receipts",
}
JUNK_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".tmp", ".bak", ".out", ".err")
GENERATED_PREFIXES = (Path("analysis/figures"),)
INTERNAL_PREFIXES = (Path(".git"),)


def generated(relative: Path) -> bool:
    return any(relative == prefix or prefix in relative.parents for prefix in GENERATED_PREFIXES)


def internal(relative: Path) -> bool:
    return any(relative == prefix or prefix in relative.parents for prefix in INTERNAL_PREFIXES)


def stable_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if generated(relative) or internal(relative):
            continue
        if JUNK_PARTS.intersection(relative.parts):
            continue
        yield path, relative


def check_json(errors: list[str]) -> int:
    count = 0
    for path, relative in stable_files():
        if path.is_file() and path.suffix == ".json":
            count += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                errors.append(f"invalid JSON {relative}: {error}")
    return count


def check_markdown_links(errors: list[str]) -> tuple[int, int]:
    documents = 0
    links = 0
    for path, relative in stable_files():
        if not path.is_file() or path.suffix.lower() not in (".md", ".markdown"):
            continue
        documents += 1
        text = path.read_text(encoding="utf-8")
        for raw in MARKDOWN_LINK.findall(text):
            target = raw.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split(maxsplit=1)[0]
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target or Path(target).is_absolute():
                continue
            links += 1
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes repository in {relative}: {raw}")
                continue
            if not resolved.exists():
                errors.append(f"broken link in {relative}: {raw}")
    return documents, links


def check_tree(errors: list[str]) -> int:
    files = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if generated(relative) or internal(relative):
            continue
        if path.is_symlink() and not path.exists():
            errors.append(f"broken symlink: {relative}")
        if path.is_file():
            files += 1
            if JUNK_PARTS.intersection(relative.parts) or path.name.endswith(JUNK_SUFFIXES):
                errors.append(f"runtime or editor artifact: {relative}")
    return files


def main() -> int:
    errors: list[str] = []
    files = check_tree(errors)
    json_files = check_json(errors)
    markdown_files, local_links = check_markdown_links(errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"repository check failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(
        "repository check passed: "
        f"{files} files, {json_files} JSON files, "
        f"{markdown_files} Markdown files, {local_links} local links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
