"""Portable process-memory measurements and compact byte formatting."""

from __future__ import annotations

import sys


def trim_allocator() -> bool:
    """Return free glibc heap pages to the OS at coarse phase boundaries."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False


def _linux_status_value(name: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith(name + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _linux_smaps_rollup() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup", encoding="ascii") as smaps:
            for line in smaps:
                fields = line.split()
                if len(fields) >= 2 and fields[0].endswith(":"):
                    values[fields[0][:-1]] = int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    return values


def current_rss_bytes() -> int:
    """Return resident memory for this process, or zero if unavailable."""
    value = _linux_status_value("VmRSS")
    if value is not None:
        return value
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return 0
    return 0


def current_pss_bytes() -> int:
    """Return this process's proportional set size, or zero if unavailable.

    PSS divides each shared page by the number of processes mapping it. The
    sum across colocated MPI ranks therefore approximates their physical node
    footprint without the shared-page double counting inherent in RSS.
    """
    return _linux_smaps_rollup().get("Pss", 0)


def current_uss_bytes() -> int:
    """Return uniquely resident private pages, or zero if unavailable."""
    values = _linux_smaps_rollup()
    return sum(
        values.get(name, 0)
        for name in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
    )


def peak_rss_bytes() -> int:
    """Return the process high-water resident set, or current RSS as fallback."""
    value = _linux_status_value("VmHWM")
    if value is not None:
        return value
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, OSError):
        return current_rss_bytes()


def available_memory_bytes() -> int:
    """Return currently available node memory where the OS exposes it."""
    try:
        with open("/proc/meminfo", encoding="ascii") as memory:
            for line in memory:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def format_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"
