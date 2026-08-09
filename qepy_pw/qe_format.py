"""Small text-format helpers shared by QE-style executables."""

from __future__ import annotations

from datetime import datetime


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
