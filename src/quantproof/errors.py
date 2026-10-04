"""Exception hierarchy.

Every error raised deliberately by QuantProof derives from :class:`QuantProofError`
and carries a message that says what went wrong *and* how to fix it.
"""

from __future__ import annotations


class QuantProofError(Exception):
    """Base class for all QuantProof errors."""


class QuantProofDataError(QuantProofError):
    """Input data cannot be used for the requested analysis."""


class QuantProofConfigError(QuantProofError):
    """Configuration is invalid or internally inconsistent."""


class QuantProofStrategyError(QuantProofError):
    """A strategy file or callable does not satisfy the strategy contract."""


class QuantProofInputError(QuantProofError, ValueError):
    """A function argument is outside its documented domain."""
