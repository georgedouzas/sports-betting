"""Implements the dataloader shared by every sport whose data comes from sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import Self

import pandas as pd

from .. import ParamGrid
from ..sources._base import BaseOddsSource, BaseStatsSource, RawItem
from ..sources._fetch import fetch_payloads
from ..sources._resolver import ALIASES, resolve
from ..sources._schema import EVENT_COLS, IDENTITY_COLS
from ._base import BaseDataLoader


class DataLoader(BaseDataLoader):
    """The dataloader of data that comes from sources.

    There is one dataloader for every sport, because the sport is a property of the sources rather than of the loader. A
    feed of soccer matches stays a feed of soccer matches whatever it is paired with, so the loader reads the sport off
    the statistics source and refuses a pairing whose odds are about a different one.

    It downloads the data into memory when you extract, and holds it on the object. Extract again and it downloads
    again, so the object always carries the latest data; keep what you have with `save`, and read it back with
    `load_dataloader`.

    Args:
        param_grid:
            Selects the seasons to train on. Keys are `'league'`, `'division'`
            and `'year'`; values are the allowed values, mirroring scikit-learn's
            `ParameterGrid`. The default `None` selects everything the sources
            publish. It bounds only the training data — the fixtures are whatever
            is upcoming in the selected leagues.

        stats:
            The source of the statistics.

        odds:
            The source of the odds. It may differ from the statistics source,
            since free statistics and paid odds are complementary. The default
            `None` gives a dataloader with no markets, for unsupervised use.

        aliases:
            The team names of the odds source, mapped to the names of the
            statistics source, for the clubs the two feeds name differently. They
            are added to the ones the library already knows.

        max_unmatched_rate:
            The proportion of matches that may go without odds when the two
            sources differ. The default `0.0` allows none, since a match whose
            odds silently go missing shrinks the dataset and skews the backtest.

    Attributes:
        stats_ (pd.DataFrame):
            The downloaded statistics snapshots: the selected seasons, plus each
            selected league's season in progress, from which the fixtures come.

        odds_ (pd.DataFrame):
            The downloaded odds snapshots of the selected provider.

        reconciliation_ (ReconciliationReport):
            How well the two sources' matches were paired. Set only when they are
            different sources.

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import FootballDataOdds, FootballDataStats
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Italy'], 'division': [1], 'year': [2024]},
        ...     stats=FootballDataStats(),
        ...     odds=FootballDataOdds(),
        ... )
        >>> # The sources say what sport it is; the loader never chose.
        >>> dataloader.sport
        'soccer'
        >>> # X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
    """

    def __init__(
        self: Self,
        param_grid: ParamGrid | None = None,
        stats: BaseStatsSource | None = None,
        odds: BaseOddsSource | None = None,
        aliases: dict[str, str] | None = None,
        max_unmatched_rate: float = 0.0,
    ) -> None:
        super().__init__(param_grid)
        self.stats = stats
        self.odds = odds
        self.aliases = aliases
        self.max_unmatched_rate = max_unmatched_rate

    def _resolved(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None]:
        """Return the statistics and odds sources, checked to be about the same sport.

        You choose where the data comes from. So a missing statistics source is an error, and pairing soccer statistics
        with basketball odds is caught here rather than deeper.
        """
        if self.stats is None:
            msg = 'No `stats` source. A dataloader does not choose where its data comes from; you do.'
            raise ValueError(msg)
        if self.odds is not None and self.stats.sport != (self.odds.sport or self.stats.sport):
            msg = (
                f'The statistics are {self.stats.sport} and the odds are {self.odds.sport}. They are about different '
                f'sports, so nothing could pair them.'
            )
            raise ValueError(msg)
        return self.stats, self.odds

    @property
    def sport(self: Self) -> str | None:
        """The sport the sources carry."""
        stats_source, _ = self._resolved()
        return stats_source.sport

    @property
    def sources(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None]:
        """The statistics and odds sources."""
        return self._resolved()

    def _catalogue(self: Self, source: BaseStatsSource | BaseOddsSource) -> list[dict]:
        """Return the combinations a source publishes for the selection.

        The source is told what was selected, so a feed whose catalogue is as large as its data reads only the part of
        it that could hold the selection. A selection of three leagues never reads the index of a fourth.
        """
        payloads = fetch_payloads(source.index_items(self.param_grid), source.request_url)
        return source.catalogue(payloads)

    def _all_params(self: Self) -> list[dict]:
        """Return the combinations both sources publish for the selection.

        With odds it is the intersection: a season the statistics carry but the odds do not cannot be bet on, so it is
        left out rather than offered and left to lose its odds silently later. With no odds the statistics are the whole
        of it.
        """
        stats_source, odds_source = self._resolved()
        stats_params = self._catalogue(stats_source)
        if odds_source is None:
            return stats_params
        priced = {tuple(sorted(params.items())) for params in self._catalogue(odds_source)}
        return [params for params in stats_params if tuple(sorted(params.items())) in priced]

    def _paired(
        self: Self,
        stats: pd.DataFrame,
        odds_items: list[RawItem],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the statistics and the odds fetched for them, paired when the two feeds differ.

        The statistics are already finalized; the odds are fetched, finalized and, when they come from a source that
        names its clubs differently, paired to the statistics.
        """
        stats_source, odds_source = self._resolved()
        if odds_source is None:
            return stats, self.no_odds()
        odds = self._finalize(odds_source.to_snapshots(fetch_payloads(odds_items, odds_source.request_url)))
        if stats_source.name != odds_source.name and not odds.empty:
            aliases = {**ALIASES, **(self.aliases or {})}
            odds, self.reconciliation_ = resolve(stats, odds, aliases, self.max_unmatched_rate)
        return stats, odds

    @staticmethod
    def _moments(matches: pd.DataFrame) -> pd.DataFrame:
        """Return the matches an odds source has to price, as identity and moment rows."""
        return matches[[*IDENTITY_COLS, *EVENT_COLS]].drop_duplicates()

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download the training statistics and odds of the selection and return their long snapshots.

        It fetches every time it is called, so the object always carries the latest data. The statistics are the seasons
        the selection names, and the odds price those same matches.
        """
        stats_source, odds_source = self._resolved()
        params = self._filter_params(self._all_params())
        stats = self._finalize(
            stats_source.to_snapshots(fetch_payloads(stats_source.required_items(params), stats_source.request_url)),
        )
        schedule = self._moments(stats) if odds_source is not None and odds_source.needs_schedule() else None
        odds_items = odds_source.required_items(params, schedule) if odds_source is not None else []
        return self._paired(stats, odds_items)

    def _fixtures_snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download the upcoming matches of the selection and return their long snapshots.

        The upcoming matches come from the season each selected league is in the middle of, whatever season was chosen
        to train on. The odds price the matches still to be played, and nothing already finished.
        """
        stats_source, odds_source = self._resolved()
        params = self._filter_params(self._all_params())
        stats = self._finalize(
            stats_source.to_snapshots(fetch_payloads(stats_source.fixtures_items(params), stats_source.request_url)),
        )
        upcoming = stats.loc[self._upcoming(stats)] if not stats.empty else stats
        schedule = self._moments(upcoming) if odds_source is not None and odds_source.needs_schedule() else None
        odds_items = odds_source.fixtures_items(params, schedule) if odds_source is not None else []
        return self._paired(stats, odds_items)
