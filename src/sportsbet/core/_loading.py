"""Load a Python object from a file-and-name reference."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ._errors import BuildError


def load_object(reference: str) -> object:
    """Return the object a reference names.

    Args:
        reference: A string with a path to a Python file and the name of an object inside it, separated by a colon.

    Returns:
        The object the reference names.

    Raises:
        BuildError: If the reference is malformed, the file is missing or unreadable, or it has no such object.
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
