"""Define the error the builders raise when their arguments do not name a valid object."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class BuildError(ValueError):
    """Raised when a builder cannot turn its arguments into the object asked for."""
