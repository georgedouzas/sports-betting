"""Implements the file a surface writes so that the data is downloaded once and reused.

Downloading is the slow part and, with a metered odds feed, the part that costs money. So one command downloads and
writes what it got, and everything after it reads that file. A surface that rebuilt from the selection each time would
buy the same seasons again on every call.

The file holds the dataloader and the training data it extracted. The dataloader on its own is enough to extract
fixtures, since it remembers what it was told, and the training data is there so that backtesting and fitting do not
have to download it again.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

import cloudpickle

from .dataloaders import BaseDataLoader

KEY = 'dataloader'


def save_dataloader(path: str, dataloader: BaseDataLoader, train: tuple) -> None:
    """Save an extracted dataloader and its training data, so it is downloaded once and reused.

    Args:
        path:
            Where to write it.
        dataloader:
            The dataloader that did the extracting.
        train:
            The training data it extracted.
    """
    with Path(path).open('wb') as file:
        cloudpickle.dump({KEY: dataloader, 'train': train}, file)


def load_dataloader(path: str) -> tuple[BaseDataLoader, tuple]:
    """Load a dataloader and its training data saved by an extract.

    A file written by `BaseDataLoader.save` holds the dataloader on its own, and is read here too, with no training
    data to go with it.

    Args:
        path:
            The file to read.

    Returns:
        loaded:
            The dataloader, and the training data it extracted.
    """
    with Path(path).open('rb') as file:
        saved: Any = cloudpickle.load(file)
    if isinstance(saved, BaseDataLoader):
        return saved, ()
    return saved[KEY], saved['train']
