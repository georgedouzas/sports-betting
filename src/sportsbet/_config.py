"""Implements the configuration that a surface is handed.

It is a Python module, and it hands over a dataloader that has already been built. Only a built one carries its sources,
and only a source carries a key, which is what lets every sport and every source reach a surface that is not Python.

The command line and the server read the same configuration through this module, so the two cannot drift. A second way
of describing a dataloader would be a second set of capabilities, which is the thing they are trying to stop being.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from .datasets import BaseDataLoader
from .evaluation import BaseBettor

OUTDATED = (
    '`DATALOADER_CLASS` and `PARAM_GRID` are no longer used. A configuration hands over a dataloader you have already '
    'built, so that it can carry its sources, and a source can carry its key:\n\n'
    "    DATALOADER = SoccerDataLoader(param_grid={'league': ['England'], 'year': [2025]})"
)


class ConfigError(ValueError):
    """Raised when a configuration does not hand over what a surface needs."""


def read_module(config_path: str) -> ModuleType:
    """Return the configuration as a module."""
    if not Path(config_path).exists():
        msg = f'The configuration file `{config_path}` does not exist.'
        raise ConfigError(msg)
    spec = spec_from_file_location('sportsbet_config', config_path)
    if spec is None or spec.loader is None:
        msg = f'The configuration file `{config_path}` could not be read as Python.'
        raise ConfigError(msg)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_dataloader(mod: ModuleType) -> BaseDataLoader:
    """Return the dataloader a configuration hands over, or say what is wrong with it."""
    if hasattr(mod, 'DATALOADER_CLASS') or hasattr(mod, 'PARAM_GRID'):
        raise ConfigError(OUTDATED)
    if not hasattr(mod, 'DATALOADER'):
        msg = 'The configuration does not have a `DATALOADER` variable.'
        raise ConfigError(msg)
    if not isinstance(mod.DATALOADER, BaseDataLoader):
        msg = '`DATALOADER` should be a dataloader that has been built, not a class.'
        raise ConfigError(msg)
    dataloader: BaseDataLoader = mod.DATALOADER
    return dataloader


def read_bettor(mod: ModuleType) -> BaseBettor:
    """Return the bettor a configuration hands over, or say what is wrong with it."""
    if not hasattr(mod, 'BETTOR'):
        msg = 'The configuration does not have a `BETTOR` variable.'
        raise ConfigError(msg)
    if not isinstance(mod.BETTOR, BaseBettor):
        msg = '`BETTOR` should be a bettor object.'
        raise ConfigError(msg)
    bettor: BaseBettor = mod.BETTOR
    return bettor
