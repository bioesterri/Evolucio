"""Cooperative host-side interruption at chunk boundaries."""

import signal
from collections.abc import Generator
from contextlib import contextmanager
from threading import current_thread, main_thread
from types import FrameType

from evolucio.persistence.local.manifest import utc_now


class StopController:
    def __init__(self) -> None:
        self.request: tuple[str, str] | None = None

    @property
    def stop_requested(self) -> bool:
        return self.request is not None

    def request_stop(self, reason: str = "user_interruption") -> None:
        if self.request is None:
            self.request = (reason, utc_now())

    @contextmanager
    def handle_sigint(self) -> Generator[None]:
        if current_thread() is not main_thread():
            yield
            return

        def interrupt(signum: int, frame: FrameType | None) -> None:
            if self.stop_requested:
                raise KeyboardInterrupt
            self.request_stop()

        previous = signal.signal(signal.SIGINT, interrupt)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)
