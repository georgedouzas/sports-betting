"""Implements how a secret is reached, which is by name.

A caller names the variable holding a secret and never carries the secret itself, exactly as an odds key is named today.
An argument is written into a shell history, a transcript and a traceback, so a secret passed as one has been published.
Named, it stays where the user put it.
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
        >>> CredentialRef('BETFAIR_APP_KEY').var
        'BETFAIR_APP_KEY'
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
