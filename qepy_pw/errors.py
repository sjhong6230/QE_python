class QEInputError(ValueError):
    """The QE input is invalid or outside the supported scalar-SCF subset."""


class UnsupportedFeatureError(NotImplementedError):
    """A valid QE feature has not yet been ported."""

