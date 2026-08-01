"""Low-overhead QE-style cumulative CPU and wall-clock timers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Iterator


@dataclass
class TimingEntry:
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    calls: int = 0


class TimingRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, TimingEntry] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        try:
            yield
        finally:
            entry = self.entries.setdefault(name, TimingEntry())
            entry.cpu_seconds += time.process_time() - cpu_start
            entry.wall_seconds += time.perf_counter() - wall_start
            entry.calls += 1

    def start(self) -> tuple[float, float]:
        return time.process_time(), time.perf_counter()

    def stop(
        self, name: str, started: tuple[float, float], calls: int = 1
    ) -> None:
        cpu_start, wall_start = started
        entry = self.entries.setdefault(name, TimingEntry())
        entry.cpu_seconds += time.process_time() - cpu_start
        entry.wall_seconds += time.perf_counter() - wall_start
        entry.calls += calls

    def snapshot(self) -> dict[str, TimingEntry]:
        return {
            name: TimingEntry(
                entry.cpu_seconds, entry.wall_seconds, entry.calls
            )
            for name, entry in self.entries.items()
        }
