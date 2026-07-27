"""Define the shared options the commands accept."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Callable

import click
from click.decorators import FC

from ..core import STATUSES
from ..dataloaders import DEFAULT_KEY_ENV, ODDS_SOURCES, STATS_SOURCES

SELECTION: list[Callable[[FC], FC]] = [
    click.option('--league', 'leagues', multiple=True, help='A league to select. Repeat it to select more.'),
    click.option('--division', 'divisions', multiple=True, type=int, help='A division to select. Repeatable.'),
    click.option('--year', 'years', multiple=True, type=int, help='A season, by the year it ends. Repeatable.'),
    click.option(
        '--stats',
        required=True,
        type=click.Choice(sorted(STATS_SOURCES)),
        help='Where the statistics come from.',
    ),
    click.option(
        '--odds',
        type=click.Choice(sorted(ODDS_SOURCES)),
        help='Where the odds come from.',
    ),
    click.option(
        '--odds-key-env',
        default=DEFAULT_KEY_ENV,
        show_default=True,
        help='The environment variable holding the odds key.',
    ),
    click.option('--odds-market', 'odds_markets', multiple=True, help='A market to price, e.g. `h2h`. Repeatable.'),
    click.option('--odds-region', 'odds_regions', multiple=True, help='A region to price, e.g. `eu`. Repeatable.'),
    click.option(
        '--odds-moment',
        'odds_moments',
        multiple=True,
        help='A moment to price, as `status:minutes`, e.g. `inplay:45`. Repeatable.',
    ),
    click.option(
        '--alias',
        'aliases',
        multiple=True,
        help='A team the sources spell differently, as `stats name=odds name`. Repeatable.',
    ),
]

HORIZON: list[Callable[[FC], FC]] = [
    click.option('--drop-na-thres', default=0.0, show_default=True, help='The threshold to drop missing columns.'),
    click.option(
        '--target-event-status',
        type=click.Choice(STATUSES),
        help='Where the targets are taken from.',
    ),
    click.option('--target-event-time', help='The moment of the targets when they are in-play, e.g. `45min`.'),
    click.option(
        '--input-event-status',
        type=click.Choice(STATUSES),
        help='The latest snapshot kept as a feature. The default keeps every one before the target.',
    ),
    click.option('--input-event-time', help='The moment of the input horizon, e.g. `45min`.'),
]

EXTRACTION: list[Callable[[FC], FC]] = [
    click.option('--odds-type', help='The odds to extract, e.g. `market_average`.'),
    *HORIZON,
]

MODEL: list[Callable[[FC], FC]] = [
    click.option(
        '--model',
        help=(
            'A scikit-learn estimator as a Python expression, as in `"OddsComparisonBettor(alpha=0.05)"` or '
            '`"ClassifierBettor(LogisticRegression(C=1.0))"`, or one you built, named by where it lives, as in '
            '`models.py:BETTOR`.'
        ),
    ),
]

BACKTEST: list[Callable[[FC], FC]] = [
    click.option('--cv', default=3, show_default=True, help='The number of time-ordered folds.'),
    click.option('--n-jobs', default=-1, show_default=True, help='The jobs the backtest runs in parallel.'),
    click.option('--verbose', default=0, show_default=True, help='How much the backtest says while it runs.'),
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

DATALOADER: list[Callable[[FC], FC]] = [
    click.option(
        '--dataloader',
        '-d',
        'dataloader_path',
        required=True,
        type=click.Path(exists=True),
        help='A dataloader saved by `dataloader train extract`.',
    ),
]


def options(*groups: list[Callable[[FC], FC]]) -> Callable[[FC], FC]:
    """Add groups of options to a command."""

    def decorate(command: FC) -> FC:
        for option in reversed([option for group in groups for option in group]):
            command = option(command)
        return command

    return decorate
