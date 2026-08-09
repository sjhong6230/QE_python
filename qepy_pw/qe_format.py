"""Small text-format helpers shared by QE-style executables."""

from __future__ import annotations

from datetime import datetime
import os


_QE_RULE = "=" + "-" * 78 + "="
_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def qe_date_and_time(now: datetime | None = None) -> tuple[str, str]:
    """Return QE's fixed-width ``cdate`` and ``ctime`` fields."""
    current = datetime.now() if now is None else now
    cdate = f"{current.day:2d}{_MONTHS[current.month - 1]}{current.year:04d}"
    ctime = f"{current.hour:2d}:{current.minute:2d}:{current.second:2d}"
    return cdate, ctime


def format_qe_duration(seconds: float, label: str) -> str:
    """Format one CPU/WALL duration with QE's primary-clock fields."""
    elapsed = max(0.0, float(seconds))
    days = int(elapsed // 86400.0)
    remainder = elapsed - 86400.0 * days
    hours = int(remainder // 3600.0)
    remainder -= 3600.0 * hours
    minutes = int(remainder // 60.0)
    remainder -= 60.0 * minutes

    if days:
        return f" {days:2d}d{hours:2d}h{minutes:2d}m {label}"
    if hours:
        return f"    {hours:2d}h{minutes:2d}m {label}"
    if minutes:
        return f" {minutes:2d}m{remainder:5.2f}s {label}"
    return f"    {remainder:5.2f}s {label}"


def format_qe_timing(program: str, cpu_seconds: float, wall_seconds: float) -> str:
    """Render an ``environment_end`` primary-clock summary line."""
    return (
        f"\n     {program:<13}: "
        f"{format_qe_duration(cpu_seconds, 'CPU')} "
        f"{format_qe_duration(wall_seconds, 'WALL')}\n"
    )


def format_qe_closing(*, success: bool = True, now: datetime | None = None) -> str:
    """Render ``closing_message`` plus QE's final job-status rule."""
    cdate, ctime = qe_date_and_time(now)
    time_str = f"This run was terminated on:  {ctime} {cdate}"
    status = "JOB DONE." if success else "JOB FAILED."
    return (
        f"\n   {time_str:<60}\n\n"
        f"{_QE_RULE}\n"
        f"   {status}\n"
        f"{_QE_RULE}\n"
    )


def _available_memory_mib() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return pages * page_size // (1024 * 1024)


def format_qe_opening(
    program: str,
    version: str,
    *,
    now: datetime | None = None,
    memory_mib: int | None = None,
) -> str:
    """Render QE's ``opening_message`` and serial environment header."""
    cdate, ctime = qe_date_and_time(now)
    available = _available_memory_mib() if memory_mib is None else memory_mib
    memory_line = (
        ""
        if available is None
        else (
            f"\n     {available:d} MiB available memory on the printing compute "
            "node when the environment starts"
        )
    )
    return (
        f"\n     Program {program} v.{version} starts on {cdate} at {ctime}\n"
        "\n"
        "     This program is part of the open-source Quantum ESPRESSO suite\n"
        "     for quantum simulation of materials; please cite\n"
        '         "P. Giannozzi et al., J. Phys.:Condens. Matter 21 395502 (2009);\n'
        '         "P. Giannozzi et al., J. Phys.:Condens. Matter 29 465901 (2017);\n'
        '         "P. Giannozzi et al., J. Chem. Phys. 152 154105 (2020);\n'
        '          URL http://www.quantum-espresso.org", \n'
        "     in publications or presentations arising from this work. More details at\n"
        "     http://www.quantum-espresso.org/quote\n"
        "\n"
        "     Serial version"
        f"{memory_line}\n\n"
    )
