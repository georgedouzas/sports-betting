"""Read the sample soccer statistics that ship with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar

from .._base import BaseStatsSource
from .._common._sample import _SampleSource


class SampleSoccerStats(_SampleSource, BaseStatsSource):
    """A frozen real season of the English and Spanish soccer first divisions, bundled with the library.

    It carries the identity of every match, each team's form before it, the half-time score and the result. The season
    is finished, so it holds training data but no fixtures.

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> source = SampleSoccerStats()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'stats', 'soccer')
        >>> # The bundled data is known up front.
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
