"""Build the dataloader and bettor a command was told to use."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from rich.console import Console
from rich.panel import Panel

from ..core import BuildError, ExecutionError
from ..dataloaders import BaseDataLoader, build_dataloader, build_extraction_settings
from ..evaluation import BaseBettor, build_bettor

SELECTED = (
    'leagues',
    'divisions',
    'years',
    'stats',
    'odds',
    'odds_key_env',
    'odds_markets',
    'odds_regions',
    'odds_moments',
    'aliases',
)
EXTRACTED = (
    'drop_na_thres',
    'odds_type',
    'target_event_status',
    'target_event_time',
    'input_event_status',
    'input_event_time',
)
MODELLED = ('model',)


def _to_list(value: object) -> object:
    """Return a repeated option as a list."""
    return list(value) if isinstance(value, tuple) else value


def _collect_given(selection: dict[str, object], names: tuple[str, ...]) -> dict[str, Any]:
    """Return what a command was told, of the things a builder asks for."""
    return cast('dict[str, Any]', {name: _to_list(selection[name]) for name in names if name in selection})


def _build_selected(selection: dict[str, object]) -> BaseDataLoader:
    """Build the dataloader a command was told to use."""
    return build_dataloader(**_collect_given(selection, SELECTED))


def _build_modelled(selection: dict[str, object]) -> BaseBettor:
    """Build the bettor a command was told to use."""
    return build_bettor(**_collect_given(selection, MODELLED))


def _build_extraction(selection: dict[str, object]) -> dict[str, Any]:
    """Return how a command was told to extract."""
    return build_extraction_settings(**_collect_given(selection, EXTRACTED))


@contextmanager
def _report_errors() -> Iterator[None]:
    """Say what went wrong."""
    try:
        yield
    except (BuildError, ExecutionError, ValueError) as error:
        Console().print(Panel.fit(f'[bold red]{error}'))
        raise SystemExit(1) from None
