"""Read the sample soccer odds that ship with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar

from .._base import BaseOddsSource
from .._common._sample import _SampleSource


class SampleSoccerOdds(_SampleSource, BaseOddsSource):
    """The odds of the soccer sample data that ships with the library.

    The market average and the market maximum of the same real season, as the free feed publishes them. They are the
    prices offered before kick-off, so a bet placed on them is a pre-match bet.

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> source = SampleSoccerOdds()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'odds', 'soccer')
        >>> dataloader = DataLoader(stats=SampleSoccerStats(), odds=source)
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
        >>> # The providers and the markets are read from the data, not registered anywhere.
        >>> dataloader.get_odds_types()
        ['market_average', 'market_maximum']
        >>> list(Y.columns)
        ['home_win__postplay__0min', 'draw__postplay__0min', 'away_win__postplay__0min', \
'over_2.5__postplay__0min', 'under_2.5__postplay__0min']
    """

    kind: ClassVar[str] = 'odds'
