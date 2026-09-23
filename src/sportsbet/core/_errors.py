"""Define the errors the library raises."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.exceptions import NotFittedError


class BuildError(ValueError):
    """Raised when a builder cannot turn its arguments into the object asked for."""


class ExecutionError(Exception):
    """Raised when placing cannot go ahead."""


class CancellationUnsupportedError(ExecutionError):
    """Raised when a venue is asked to cancel and cannot."""


class VenueBlockedError(ExecutionError):
    """Raised when a venue blocks automated access."""


class CredentialError(ExecutionError):
    """Raised when a named variable holds nothing."""


class NotExtractedError(NotFittedError):
    """Raised where a dataloader is asked for data that a previous extraction has to fix first.

    Examples:
        >>> from sportsbet.core import NotExtractedError
        >>> from sportsbet.dataloaders import build_dataloader
        >>> dataloader = build_dataloader(stats='football-data', leagues=['Italy'], years=[2024])
        >>> try:
        ...     dataloader.extract_fixtures_data()
        ... except NotExtractedError as error:
        ...     print(error)
        Call `extract_train_data` before `extract_fixtures_data`, since it fixes the columns to match.
    """
