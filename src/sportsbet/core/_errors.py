"""Define the error the builders raise when their arguments do not name a valid object."""

from __future__ import annotations


class BuildError(ValueError):
    """Raised when a builder cannot turn its arguments into the object asked for."""
