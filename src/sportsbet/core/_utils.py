"""Core utilities."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

from ._errors import BuildError


def format_event_time(event_time: pd.Timedelta) -> str:
    """Render an event time as the whole-minute token used in column names.

    Args:
        event_time: A time delta, e.g. `pd.Timedelta('60min')`.

    Returns:
        The token, e.g. `60min`.
    """
    total_minutes = int(event_time.total_seconds() / 60)
    return f'{total_minutes}min'


def parse_event_time(token: str) -> pd.Timedelta:
    """Read the whole-minute token used in column names back into a time delta.

    Args:
        token: A whole-minute token, e.g. `60min`.

    Returns:
        The time delta the token names.
    """
    return pd.Timedelta(minutes=int(token[: -len('min')]))


def load_object(reference: str) -> object:
    """Return the object a reference names.

    Args:
        reference: A string with a path to a Python file and the name of an object inside it, separated by a colon.

    Returns:
        The object the reference names.
    """
    path, _, name = reference.rpartition(':')
    if not name:
        msg = f'`{reference}` should name an object inside a Python file, as in `models.py:bettor`.'
        raise BuildError(msg)
    if not Path(path).exists():
        msg = f'The file `{path}` does not exist.'
        raise BuildError(msg)
    spec = spec_from_file_location('sportsbet_model', path)
    if spec is None or spec.loader is None:
        msg = f'The file `{path}` could not be read as Python.'
        raise BuildError(msg)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, name):
        msg = f'The file `{path}` has no `{name}` in it.'
        raise BuildError(msg)
    return getattr(mod, name)
