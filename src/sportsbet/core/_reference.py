"""Load the object a reference names."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ._errors import BuildError


def _load_object(reference: str) -> object:
    """Return the object a reference names, which is a Python file and a name inside it."""
    path, _, name = reference.rpartition(':')
    if not name:
        msg = f'`{reference}` should name an object inside a Python file, as in `models.py:BETTOR`.'
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
