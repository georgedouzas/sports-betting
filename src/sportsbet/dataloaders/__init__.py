"""Shape the source data for modelling."""

from ._base import BaseDataLoader, load_dataloader
from ._extraction import build_extraction_settings
from ._factory import DEFAULT_KEY_ENV, ODDS_SOURCES, STATS_SOURCES, build_dataloader
from ._sourced import DataLoader

__all__: list[str] = [
    'DEFAULT_KEY_ENV',
    'ODDS_SOURCES',
    'STATS_SOURCES',
    'BaseDataLoader',
    'DataLoader',
    'build_dataloader',
    'build_extraction_settings',
    'load_dataloader',
]
