"""Load the modelling data of every sport whose data comes from sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Self

import pandas as pd

from ..core import EVENT_COLS, IDENTITY_COLS, ParamGrid
from ..sources import BaseOddsSource, BaseStatsSource, RawItem, resolve_odds
from ._base import BaseDataLoader


class DataLoader(BaseDataLoader):
    """Download source data and shape it into moment-aware modelling data.

    There is one dataloader for every sport. The loader reads the sport off the statistics source and pairs it only with
    odds about the same sport.

    It downloads the data into memory when you extract, and holds it on the object. Extract again and it downloads
    again. Keep what you have with `save`, and read it back with `load_dataloader`.

    Args:
        param_grid:
            Selects the seasons to train on. Keys are `'league'`, `'division'`
            and `'year'`, and values are the allowed values, mirroring
            scikit-learn's `ParameterGrid`. The default `None` selects everything
            the sources publish. It bounds only the training data. The fixtures
            are whatever is upcoming in the selected leagues.

        stats:
            The source of the statistics.

        odds:
            The source of the odds, which may differ from the statistics source.
            The default `None` gives a dataloader with no markets, for
            `extract_exploration_data`.

        aliases:
            The team names of the odds source, mapped to the names of the
            statistics source, for the clubs the two feeds name differently. They
            are added to the ones the library already knows.

    Attributes:
        stats_ (pd.DataFrame):
            The downloaded statistics snapshots: the selected seasons, plus each
            selected league's season in progress.

        odds_ (pd.DataFrame):
            The downloaded odds snapshots of the selected provider.

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import FootballDataOdds, FootballDataStats
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Italy'], 'division': [1], 'year': [2024]},
        ...     stats=FootballDataStats(),
        ...     odds=FootballDataOdds(),
        ... )
        >>> # The sources say what sport it is.
        >>> dataloader.sport_
        'soccer'
        >>> # X, Y, O = dataloader.extract_train_data(odds_type='market_maximum')
    """

    def __init__(
        self: Self,
        param_grid: ParamGrid | None = None,
        stats: BaseStatsSource | None = None,
        odds: BaseOddsSource | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        super().__init__(param_grid)
        self.stats = stats
        self.odds = odds
        self.aliases = aliases

    def _resolve_sources(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None]:
        """Return the statistics and odds sources, checked to be about the same sport."""
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
    def sport_(self: Self) -> str | None:
        """The sport the sources carry."""
        stats_source, _ = self._resolve_sources()
        return stats_source.sport

    @property
    def sources_(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None]:
        """The statistics and odds sources."""
        return self._resolve_sources()

    def _read_catalogue(self: Self, source: BaseStatsSource | BaseOddsSource) -> list[dict]:
        """Return the combinations a source publishes for the selection."""
        payloads = source.fetch_items(source.list_index_items(self.param_grid))
        return source.read_catalogue(payloads)

    def _list_all_params(self: Self) -> list[dict]:
        """Return the combinations both sources publish for the selection."""
        stats_source, odds_source = self._resolve_sources()
        stats_params = self._read_catalogue(stats_source)
        if odds_source is None:
            return stats_params
        priced = {tuple(sorted(params.items())) for params in self._read_catalogue(odds_source)}
        return [params for params in stats_params if tuple(sorted(params.items())) in priced]

    def _fetch_paired_odds(
        self: Self,
        stats: pd.DataFrame,
        odds_items: list[RawItem],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the statistics and the odds fetched for them, paired when the two feeds differ."""
        stats_source, odds_source = self._resolve_sources()
        if odds_source is None:
            return stats, self._build_empty_odds()
        odds = self._finalize(odds_source.to_snapshots(odds_source.fetch_items(odds_items)))
        if stats_source.name != odds_source.name and not odds.empty:
            odds = resolve_odds(stats, odds, self.aliases)
        return stats, odds

    @staticmethod
    def _select_moments(matches: pd.DataFrame) -> pd.DataFrame:
        """Return the matches an odds source has to price, as identity and moment rows."""
        return matches[[*IDENTITY_COLS, *EVENT_COLS]].drop_duplicates()

    def _load_snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download the training statistics and odds of the selection and return their long snapshots."""
        stats_source, odds_source = self._resolve_sources()
        params = self._filter_params(self._list_all_params())
        stats = self._finalize(
            stats_source.to_snapshots(stats_source.fetch_items(stats_source.list_required_items(params))),
        )
        schedule = self._select_moments(stats) if odds_source is not None and odds_source.needs_schedule() else None
        odds_items = odds_source.list_required_items(params, schedule) if odds_source is not None else []
        return self._fetch_paired_odds(stats, odds_items)

    def _load_fixtures_snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download the upcoming matches of the selection and return their long snapshots."""
        stats_source, odds_source = self._resolve_sources()
        params = self._filter_params(self._list_all_params())
        stats = self._finalize(
            stats_source.to_snapshots(stats_source.fetch_items(stats_source.list_fixtures_items(params))),
        )
        upcoming = stats.loc[self._is_upcoming(stats)] if not stats.empty else stats
        schedule = self._select_moments(upcoming) if odds_source is not None and odds_source.needs_schedule() else None
        odds_items = odds_source.list_fixtures_items(params, schedule) if odds_source is not None else []
        return self._fetch_paired_odds(stats, odds_items)
