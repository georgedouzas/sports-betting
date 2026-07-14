"""Implements the sources of the sample data that ships with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
from sklearn.model_selection import ParameterGrid

from .. import ParamGrid
from ._base import BaseOddsSource, BaseStatsSource, RawItem, RawPayload

DATA = Path(__file__).parent / 'data'
PARAMS: ParamGrid = {'league': ['England', 'Spain'], 'division': [1], 'year': [2024]}


class _SampleSource:
    """The half of a sample source that does not depend on what it carries.

    The data ships with the library, so the catalogue is known without reading anything and every item is a file on your
    own disk. Nothing else about it is special: the store fetches it, keeps it and skips it next time, and the sport,
    the markets and the moments are read from the data exactly as they are for a feed on the internet.
    """

    name: ClassVar[str] = 'sample_soccer'
    kind: ClassVar[str]
    sport: ClassVar[str | None] = 'soccer'

    def index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return no items, since what the library ships is known without reading it."""
        return []

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the leagues, divisions and seasons the sample carries."""
        return list(ParameterGrid(PARAMS))

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the bundled file of every selected season."""
        items = []
        for param in params:
            key = f'{param["league"]}_{param["division"]}_{param["year"]}_{self.kind}'
            path = DATA / f'{key}.csv.gz'
            if path.exists():
                items.append(RawItem(source=self.name, key=key, url=path.as_uri()))
        return items

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Return the long snapshots of the bundled files."""
        if not payloads:
            return pd.DataFrame()
        frames = [pd.read_csv(io.BytesIO(payload.content), compression='gzip') for payload in payloads]
        snapshots = pd.concat(frames, ignore_index=True)
        snapshots['date'] = pd.to_datetime(snapshots['date'], utc=True)
        snapshots['event_time'] = pd.to_timedelta(snapshots['event_time'])
        return snapshots


class SampleSoccerStats(_SampleSource, BaseStatsSource):
    """The statistics of the soccer sample data that ships with the library.

    It is a real season of the English and Spanish first divisions, taken from
    [football-data.co.uk](https://www.football-data.co.uk) and frozen, carrying the identity of every match, the form of
    the two teams before it, the score at half time and the result. It needs no key and it reaches no network, so it
    runs offline, which is what makes it the data of the examples and the tests.

    The season is finished, so it has **no fixtures**. A fixture is a match that has not been played, and there are none
    left in a season that is over. To bet on something, use a live source such as
    [`FootballDataStats`][sportsbet.sources.FootballDataStats].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats
        >>> source = SampleSoccerStats()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'stats', 'soccer')
        >>> # It ships with the library, so it knows what it carries without reading anything.
        >>> source.available_params()
        [{'division': 1, 'league': 'England', 'year': 2024}, {'division': 1, 'league': 'Spain', 'year': 2024}]
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['England']},
        ...     stats=source,
        ...     odds=SampleSoccerOdds(),
        ... )
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_average', download=True)
        >>> # A whole season of the Premier League.
        >>> len(X)
        380
    """

    kind: ClassVar[str] = 'stats'


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
        >>> X, Y, O = dataloader.extract_train_data(odds_type='market_maximum', download=True)
        >>> # The providers and the markets are read from the data, not registered anywhere.
        >>> dataloader.get_odds_types()
        ['market_average', 'market_maximum']
        >>> list(Y.columns)
        ['home_win__postplay__0min', 'draw__postplay__0min', 'away_win__postplay__0min', \
'over_2.5__postplay__0min', 'under_2.5__postplay__0min']
    """

    kind: ClassVar[str] = 'odds'
