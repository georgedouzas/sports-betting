"""Read the sample soccer statistics that ship with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar

from .._base import BaseStatsSource
from .._common._sample import _SampleSource


class SampleSoccerStats(_SampleSource, BaseStatsSource):
    """The statistics of the soccer sample data that ships with the library.

    It is a real season of the English and Spanish first divisions, taken from
    [football-data.co.uk](https://www.football-data.co.uk) and frozen, carrying the identity of every match, the form of
    the two teams before it, the score at half time and the result. It needs no key and runs offline, which is what
    makes it the data of the examples and the tests.

    The season is finished, so every match in it is played: it gives training data. A fixture is a match that has not
    been played, and a finished season has none. To bet on something, use a live source such as
    [`FootballDataStats`][sportsbet.sources.FootballDataStats].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> source = SampleSoccerStats()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'stats', 'soccer')
        >>> # It ships with the library, so it knows what it carries without reading anything.
        >>> source.list_available_params()
        [{'division': 1, 'league': 'England', 'year': 2024}, {'division': 1, 'league': 'Spain', 'year': 2024}]
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['England']},
        ...     stats=source,
        ...     odds=SampleSoccerOdds(),
        ... )
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_average')
        >>> # A whole season of the Premier League.
        >>> len(X)
        380
    """

    kind: ClassVar[str] = 'stats'
