"""Module that contains the options of the CLI.

A command is told what to do in its own arguments. Nothing is written down first, because a dataloader is a sport, what
to select from it, and where its data comes from, and all of those are names.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable

import click
from click.decorators import FC

from .._selection import DEFAULT_KEY_ENV, MODELS, ODDS_SOURCES, SPORTS, STATS_SOURCES

SELECTION: list[Callable[[FC], FC]] = [
    click.option(
        '--sport',
        required=True,
        type=click.Choice(sorted(SPORTS)),
        help='The sport. `dummy` is a made-up soccer league, so the commands can be tried without downloading.',
    ),
    click.option('--league', 'leagues', multiple=True, help='A league to select. Repeat it to select more.'),
    click.option('--division', 'divisions', multiple=True, type=int, help='A division to select. Repeatable.'),
    click.option('--year', 'years', multiple=True, type=int, help='A season, by the year it ends. Repeatable.'),
    click.option(
        '--stats',
        type=click.Choice(sorted(STATS_SOURCES)),
        help="Where the statistics come from. The default is the sport's own.",
    ),
    click.option(
        '--odds',
        type=click.Choice(sorted(ODDS_SOURCES)),
        help="Where the odds come from. The default is the sport's own, and basketball has no free one.",
    ),
    click.option(
        '--odds-key-env',
        default=DEFAULT_KEY_ENV,
        show_default=True,
        help='The environment variable holding the odds key, so the key itself never appears in the command.',
    ),
    click.option('--odds-market', 'odds_markets', multiple=True, help='A market to price, e.g. `h2h`. Repeatable.'),
    click.option('--odds-region', 'odds_regions', multiple=True, help='A region to price, e.g. `eu`. Repeatable.'),
]

EXTRACTION: list[Callable[[FC], FC]] = [
    click.option('--odds-type', help='The odds to extract, e.g. `market_average`.'),
    click.option('--drop-na-thres', default=0.0, show_default=True, help='The threshold to drop missing columns.'),
    click.option('--target-event-status', help='Where the targets are taken: `preplay`, `inplay` or `postplay`.'),
    click.option('--target-event-time', help='The in-play moment of the targets, e.g. `45min`.'),
]

MODEL: list[Callable[[FC], FC]] = [
    click.option(
        '--model',
        required=True,
        help=(
            f'A ready-made model ({", ".join(MODELS)}), or a scikit-learn one you built yourself, named by where it '
            f'lives, as in `models.py:BETTOR`.'
        ),
    ),
    click.option('--alpha', default=0.05, show_default=True, help='The tolerance of `odds-comparison`.'),
    click.option('--init-cash', type=float, help='The cash a backtest starts with.'),
    click.option('--stake', type=float, help='The stake of each bet.'),
    click.option('--betting-market', 'betting_markets', multiple=True, help='A market to bet on. Repeatable.'),
    click.option('--cv', default=3, show_default=True, help='The number of time-ordered folds of a backtest.'),
    click.option('--n-jobs', default=-1, show_default=True, help='The jobs a backtest runs in parallel.'),
    click.option('--verbose', default=0, show_default=True, help='How much a backtest says while it runs.'),
]

OUTPUT: list[Callable[[FC], FC]] = [
    click.option(
        '--output',
        '-o',
        'data_path',
        type=click.Path(),
        help='A directory to write the results to, as CSV.',
    ),
]


def options(*groups: list[Callable[[FC], FC]]) -> Callable[[FC], FC]:
    """Add groups of options to a command."""

    def decorate(command: FC) -> FC:
        for option in reversed([option for group in groups for option in group]):
            command = option(command)
        return command

    return decorate
