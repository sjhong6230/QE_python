"""Low-overhead QE-style cumulative CPU and wall-clock timers."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class TimingEntry:
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    calls: int = 0


class _Measurement:
    """Small context manager avoiding generator allocation in hot kernels."""

    __slots__ = ("registry", "name", "cpu_start", "wall_start")

    def __init__(self, registry: "TimingRegistry", name: str) -> None:
        self.registry = registry
        self.name = name
        self.cpu_start = 0.0
        self.wall_start = 0.0

    def __enter__(self) -> None:
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        entry = self.registry.entries.setdefault(self.name, TimingEntry())
        entry.cpu_seconds += time.process_time() - self.cpu_start
        entry.wall_seconds += time.perf_counter() - self.wall_start
        entry.calls += 1


class TimingRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, TimingEntry] = {}

    def measure(self, name: str) -> _Measurement:
        return _Measurement(self, name)

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
