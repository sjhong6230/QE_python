from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QEWarning:
    """A non-fatal diagnostic rendered like QE's ``infomsg`` output."""

    routine: str
    message: str


class QEDiagnosticError(Exception):
    """Base class carrying the fields printed by QE's ``errore`` routine."""

    def __init__(
        self,
        message: str,
        *,
        routine: str = "pw.py",
        code: int = 1,
    ) -> None:
        super().__init__(message)
        self.routine = routine
        self.code = code


class QEInputError(QEDiagnosticError, ValueError):
    """The QE input is invalid for the implemented scalar-SCF path."""


class UnsupportedFeatureError(QEDiagnosticError, NotImplementedError):
    """A valid QE feature has not yet been ported."""


def not_implemented(feature: str) -> str:
    """Make unsupported-feature diagnostics unambiguous to QE users."""

    return f"{feature} is not implemented in PWSCF-PY"


def format_qe_error(error: BaseException) -> str:
    """Render an exception using QE 7.5 ``error_handler.f90`` formatting."""

    routine = getattr(error, "routine", "pw.py")
    code = int(getattr(error, "code", 1))
    border = " %" + "%" * 77
    return (
        f"\n{border}\n"
        f"     Error in routine {routine} ({code}):\n"
        f"     {error}\n"
        f"{border}\n\n"
        "     stopping ...\n"
    )


def format_qe_warning(warning: QEWarning) -> str:
    """Render a warning using QE 7.5 ``infomsg`` formatting."""

    return (
        f"     Message from routine {warning.routine}:\n"
        f"     {warning.message}\n"
    )

