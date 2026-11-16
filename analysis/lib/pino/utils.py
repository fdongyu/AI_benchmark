"""Runtime utilities shared by training and adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import hashlib
import json
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str, require_cuda: bool) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "CUDA was requested but is unavailable; refusing silent CPU fallback"
            )
        return torch.device("cpu")
    return device


def git_revision(directory: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace a JSON-lines file with complete records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: list[str | Path]) -> str:
    """Hash names and contents of an ordered source-file collection."""

    digest = hashlib.sha256()
    for value in sorted((Path(path).resolve() for path in paths), key=str):
        digest.update(value.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(value)))
    return digest.hexdigest()


def sha256_manifest_split(path: str | Path, split: str) -> str:
    """Hash canonical manifest rows belonging to one immutable data split."""

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
            if str(row.get("split", "")).strip() == split
        ]
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
