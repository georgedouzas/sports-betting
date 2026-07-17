"""Read a secret from the variable named for it, never from an argument.

An argument reaches a shell history, a transcript and a traceback, so a named variable is read where it is used and
travels nowhere.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import os
from dataclasses import dataclass

from ._base import ExecutionError


class CredentialError(ExecutionError):
    """Raised when a named variable holds nothing."""


@dataclass(frozen=True)
class CredentialRef:
    """The name of a variable holding a secret.

    Examples:
        >>> from sportsbet.execution import CredentialRef
        >>> CredentialRef('VENUE_API_KEY').var
        'VENUE_API_KEY'
    """

    var: str

    def __str__(self: CredentialRef) -> str:
        """Return the name, since the value is not this object's to show."""
        return self.var


def resolve(ref: CredentialRef) -> str:
    """Return what a named variable holds, raising when it holds nothing.

    Args:
        ref:
            The name of the variable.

    Returns:
        secret:
            What the variable holds.
    """
    secret = os.environ.get(ref.var)
    if not secret:
        msg = f'`{ref.var}` is not set. Set it to the secret, or name another variable.'
        raise CredentialError(msg)
    return secret
