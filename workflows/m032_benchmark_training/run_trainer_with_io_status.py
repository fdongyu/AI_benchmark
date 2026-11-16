#!/usr/bin/env python3
"""Run the immutable trainer and classify only transient CFS I/O failures."""

from __future__ import annotations

import errno
import re
import runpy
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence


TRANSIENT_CFS_EXIT = 89
TRANSIENT_ERRNOS = frozenset(
    value
    for value in (
        errno.EIO,
        errno.ESTALE,
        errno.ETIMEDOUT,
        getattr(errno, "EREMOTEIO", None),
    )
    if value is not None
)
TRANSIENT_ERRNO_TEXT = re.compile(
    r"(?:\[Errno\s+|errno\s*=\s*)(5|110|116|121)\b"
)
TRANSIENT_MESSAGES = (
    "Input/output error",
    "Stale file handle",
    "Connection timed out",
    "Remote I/O error",
)


def is_transient_cfs_error(error: BaseException) -> bool:
    """Recognize propagated GPFS/HDF5 failures without masking model errors."""

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in TRANSIENT_ERRNOS:
            return True
        message = str(current)
        if TRANSIENT_ERRNO_TEXT.search(message) and any(
            marker in message for marker in TRANSIENT_MESSAGES
        ):
            return True
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


def run_trainer(trainer_path: Path, trainer_args: Sequence[str]) -> None:
    trainer_path = trainer_path.resolve(strict=True)
    sys.argv = [str(trainer_path), *trainer_args]
    runpy.run_path(str(trainer_path), run_name="__main__")


def main(argv: Optional[Sequence[str]] = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        raise SystemExit("usage: run_trainer_with_io_status.py TRAINER [TRAINER_ARGS ...]")
    try:
        run_trainer(Path(values[0]), values[1:])
    except OSError as error:
        if not is_transient_cfs_error(error):
            raise
        traceback.print_exc()
        print(
            "m032-depth4-v1: classified a transient CFS/HDF5 I/O failure; "
            f"exiting {TRANSIENT_CFS_EXIT} for a clean process retry",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(TRANSIENT_CFS_EXIT) from error


if __name__ == "__main__":
    main()
