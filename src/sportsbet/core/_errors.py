"""Define the error the builders raise when a reference cannot be built."""

from __future__ import annotations


class BuildError(ValueError):
    """Raised when the given names do not describe something that can be built."""
