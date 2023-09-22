"""Module that contains the options of the CLI."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable

import click
from click.decorators import FC


def get_data_path_option() -> Callable[[FC], FC]:
    """Get the data path option."""
    return click.option(
        '--data-path',
        '-d',
        nargs=1,
        required=False,
        default=None,
        type=str,
        help='The path to save the artifacts of betting CLI.',
    )


def get_config_path_option() -> Callable[[FC], FC]:
    """Get the configuration file path option."""
    return click.option(
        '--config-path',
        '-c',
        nargs=1,
        required=True,
        type=str,
        help='The path of the configuration file.',
    )
