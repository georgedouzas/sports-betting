"""Save and load the dataloader and its training data, so the data is downloaded once and reused."""

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
