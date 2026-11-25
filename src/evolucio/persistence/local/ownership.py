"""Nonblocking OS ownership of a local run directory."""

import errno
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

WriterStatus = Literal["active", "inactive", "indeterminate"]


@contextmanager
def run_ownership(run_dir: Path, *, create: bool = False) -> Generator[WriterStatus]:
    path = run_dir / ".writer.lock"
    if path.is_symlink():
        yield "indeterminate"
        return
    try:
        stream = path.open("a+b" if create else "r+b")
    except OSError:
        yield "indeterminate"
        return
    with stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            yield "active" if exc.errno in (errno.EACCES, errno.EAGAIN) else "indeterminate"
            return
        yield "inactive"
