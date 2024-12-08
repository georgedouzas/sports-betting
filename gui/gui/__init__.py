"""Includes all pages."""

from .pages import index
from .pages.data import dataloader, params, train

__all__: list[str] = [
    'index',
    'dataloader',
    'params',
    'train',
]
