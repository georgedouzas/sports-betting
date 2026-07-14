"""Implements the dataloader shared by every sport whose data comes from sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from itertools import product
from typing import Self

import pandas as pd
from rich.console import Console
from rich.progress import Progress

from .. import ParamGrid
from ..sources._base import BaseOddsSource, BaseSource, BaseStatsSource, RawItem, RawPayload
from ..sources._resolver import ALIASES, resolve
from ..sources._schema import EVENT_COLS, IDENTITY_COLS
from ..sources._store import BaseStore, LocalStore, NotPreparedError, PreparationReport, payloads_digest
from ._base import PARAM_COLS, BaseDataLoader


class DataLoader(BaseDataLoader):
    """The dataloader of data that comes from sources.

    There is one of these, not one per sport. The store, the download, the schedule and the reconciliation are the same
    whatever is being played, and nothing in them ever asks what the sport is. The sport is a property of the sources: a
    feed of soccer matches is a feed of soccer matches, and pairing it with the odds of a basketball league is refused
    rather than left to fail somewhere deeper.

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.

        stats:
            The source of the statistics. Each source carries its own settings,
            so they never spread onto the dataloader.

        odds:
            The source of the odds. It may differ from the statistics source,
            since free statistics and paid odds are complementary rather than
            alternatives.

        store:
            Where the downloaded data is kept. The default `None` keeps it in
            `~/.sportsbet`.

        aliases:
            The team names of the odds source, mapped to the names of the
            statistics source, for when the two do not name a club the same way.
            They are added to the ones the library already knows.

        max_unmatched_rate:
            The proportion of matches that may go without odds when the two
            sources are different. The default `0.0` allows none, since a match
            whose odds are missing does not look like an error: it looks like a
            smaller dataset and a backtest that is confidently wrong.

    Attributes:
        reconciliation_ (ReconciliationReport):
            How well the matches of the two sources were found in each other.
            Only set when they are different sources.

    Examples:
        >>> import tempfile
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import FootballDataOdds, FootballDataStats, LocalStore, NotPreparedError
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Italy'], 'division': [1], 'year': [2024]},
        ...     stats=FootballDataStats(),
        ...     odds=FootballDataOdds(),
        ...     store=LocalStore(tempfile.mkdtemp()),
        ... )
        >>> # It never chose where its data comes from, so it can say what sport it is: the sources say.
        >>> dataloader.sport
        'soccer'
        >>> # Nothing reaches the network unless `download` says so, so an extraction cannot spend by accident.
        >>> try:
        ...     dataloader.extract_train_data(odds_type='market_maximum')
        ... except NotPreparedError as error:
        ...     print(error)
        The data has not been downloaded. Pass `download=True` to get it.
        >>> # X, Y, O = dataloader.extract_train_data(odds_type='market_maximum', download=True)
    """

    def __init__(
        self: Self,
        param_grid: ParamGrid | None = None,
        stats: BaseStatsSource | None = None,
        odds: BaseOddsSource | None = None,
        store: BaseStore | None = None,
        aliases: dict[str, str] | None = None,
        max_unmatched_rate: float = 0.0,
    ) -> None:
        super().__init__(param_grid)
        self.stats = stats
        self.odds = odds
        self.store = store
        self.aliases = aliases
        self.max_unmatched_rate = max_unmatched_rate

    def _resolved(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None, BaseStore]:
        """Return the sources and the store, built once and kept.

        Nothing is assumed. Which feed the data came from decides what is in it, what it costs and whether anyone may
        redistribute it, and a dataloader that chose one for you would be answering that on your behalf.
        """
        if self._components is None:
            if self.stats is None:
                msg = 'No `stats` source. A dataloader does not choose where its data comes from; you do.'
                raise ValueError(msg)
            if self.odds is not None and self.stats.sport != (self.odds.sport or self.stats.sport):
                msg = (
                    f'The statistics are {self.stats.sport} and the odds are {self.odds.sport}. They are not about the '
                    f'same matches, so nothing could pair them.'
                )
                raise ValueError(msg)
            store = self.store if self.store is not None else LocalStore()
            self._components = (self.stats, self.odds, store)
        return self._components

    @property
    def sport(self: Self) -> str | None:
        """The sport the sources carry, which is what they are of rather than what they were told to be."""
        stats_source, _, _ = self._resolved()
        return stats_source.sport

    @property
    def sources(self: Self) -> tuple[BaseStatsSource, BaseOddsSource | None]:
        """The resolved statistics and odds sources."""
        stats_source, odds_source, _ = self._resolved()
        return stats_source, odds_source

    def _authorize(self: Self, item: RawItem) -> str:
        """Return the URL of an item, authorized by the source that declared it."""
        stats_source, odds_source = self.sources
        source = odds_source if odds_source is not None and item.source == odds_source.name else stats_source
        return source.request_url(item)

    @staticmethod
    def _unique(items: list[RawItem]) -> list[RawItem]:
        """Return the items without duplicates, so an item two sources share is fetched once."""
        return list(dict.fromkeys(items))

    def _catalogue(self: Self, fetch: bool, refresh: bool = False) -> list[RawPayload]:
        """Return the payloads of the catalogue, fetching them only when allowed to.

        The whole catalogue is read, not just the selected part of it. `param_grid` says what to train on, and the
        matches that have not been played are not restricted by it: they are in the season in progress of whatever
        league they belong to, and the catalogue is what says which season that is. A feed that only knew about the
        leagues you selected could not offer you a fixture in any other one.
        """
        stats_source, odds_source, store = self._resolved()
        odds_items = odds_source.index_items() if odds_source is not None else []
        items = self._unique(stats_source.index_items() + odds_items)
        if fetch:
            held = [] if refresh else store.held(items)
            store.fetch([item for item in items if item not in held], self._authorize)
        return store.read(items)

    def _params(self: Self, fetch: bool, refresh: bool = False) -> list[dict]:
        """Return the combinations the sources publish, reading the catalogue from the store.

        With odds it is the intersection, not the union: a season whose statistics exist but whose odds do not cannot be
        bet on, so it is never selected. Asking only the statistics source would offer it and let the missing odds
        surface as a silently smaller dataset. With no odds there is nothing to intersect with, and the statistics are
        the whole of it.
        """
        stats_source, odds_source, _ = self._resolved()
        payloads = self._catalogue(fetch, refresh)
        stats_params = stats_source.catalogue([p for p in payloads if p.item.source == stats_source.name])
        if odds_source is None:
            return stats_params
        odds_params = odds_source.catalogue([p for p in payloads if p.item.source == odds_source.name])
        available = {tuple(sorted(params.items())) for params in odds_params}
        return [params for params in stats_params if tuple(sorted(params.items())) in available]

    def _all_params(self: Self) -> list[dict]:
        """Return the combinations both sources publish, used only to filter `param_grid`.

        Discovery is not an extraction, so it reads the catalogue from the feed when the store does not hold it. The
        catalogue is free.
        """
        return self._params(fetch=True)

    def _schedule(self: Self, stats_items: list[RawItem], fetch: bool, refresh: bool) -> pd.DataFrame | None:
        """Return the matches an odds source has to price, with their kick-off instants.

        An odds source that addresses its prices by instant needs it, since a season alone does not say when its matches
        are played. It comes from the statistics, which are read first, so a metered odds source can be counted exactly
        without spending anything.

        It carries the moments of the statistics as well as their kick-offs, so an odds source never buys a price for a
        moment there is nothing to pair it with. A sport whose statistics stop at the whistle has no use for the odds at
        half time, and those cost as much as the ones it can use.

        The statistics carry the season in progress of every league, since that is what a fixture's form is made of. A
        price is not needed for all of it: what is bet on is the matches selected for training and the matches not yet
        played. Anything else would be a price bought for a match nobody asked about.
        """
        stats_source, _, store = self._resolved()
        if fetch:
            held = [] if refresh else store.held(stats_items)
            store.fetch([item for item in stats_items if item not in held], stats_source.request_url)
        snapshots = self._derive(stats_source, store.read(stats_items), store)
        if snapshots.empty:
            return None
        played = pd.MultiIndex.from_frame(
            snapshots.loc[snapshots['event_status'].eq('postplay'), list(IDENTITY_COLS)],
        )
        matches = pd.MultiIndex.from_frame(snapshots[list(IDENTITY_COLS)])
        wanted = self._selected_mask(snapshots) | ~pd.Series(data=matches.isin(played), index=snapshots.index)
        return snapshots.loc[wanted, [*IDENTITY_COLS, *EVENT_COLS]].drop_duplicates()

    def _items(
        self: Self,
        fetch: bool,
        refresh: bool = False,
        available: list[dict] | None = None,
    ) -> tuple[list[RawItem], list[RawItem]]:
        """Return the items the selected parameters need from each source.

        An odds source that has to be told when the matches are is planned only after the statistics have been read, so
        the schedule it is given is the real one rather than a guess.
        """
        stats_source, odds_source, _ = self._resolved()
        available = self._params(fetch, refresh) if available is None else available
        params = self._filter_params(available)
        stats_items = stats_source.required_items(params)
        if odds_source is None:
            return stats_items, []
        schedule = self._schedule(stats_items, fetch, refresh) if odds_source.needs_schedule() else None
        return stats_items, odds_source.required_items(params, schedule)

    def _report(self: Self, fetch: bool, refresh: bool = False) -> PreparationReport:
        """Return what a preparation would fetch, what is held, and what it would cost.

        The catalogue is resolved once and handed to both the items and the unavailable parameters. Each used to resolve
        it for itself, and the index of a source is volatile and so is never held, which meant a single preparation read
        it twice and paid for it twice.
        """
        _, _, store = self._resolved()
        available = self._params(fetch, refresh)
        stats_items, odds_items = self._items(fetch, refresh, available)
        items = self._unique(stats_items + odds_items)
        held = [] if refresh else store.held(items)
        return PreparationReport(
            to_fetch=[item for item in items if item not in held],
            held=held,
            unavailable=self._unavailable(available),
        )

    def _unavailable(self: Self, available: list[dict]) -> list[dict]:
        """Return the fully specified parameters the sources do not publish."""
        if self.param_grid is None:
            return []
        published = {tuple(sorted(params.items())) for params in available}
        grids = self.param_grid if isinstance(self.param_grid, list) else [self.param_grid]
        requested = [
            dict(zip(grid, values, strict=True))
            for grid in grids
            if sorted(grid) == sorted(PARAM_COLS)
            for values in product(*[grid[key] for key in grid])
        ]
        return [params for params in requested if tuple(sorted(params.items())) not in published]

    def _download(self: Self, refresh: bool) -> None:
        """Download the data the selected parameters need.

        Only what the store does not already hold is fetched, so it is incremental and resumable. A finished season is
        not fetched again, since it is not expected to change. Upstream can still correct one, though, so nothing it
        publishes is truly immutable, and `refresh` reads everything again. That costs a metered source the same as the
        first time, which is why it is asked for rather than done on a schedule.
        """
        _, _, store = self._resolved()
        report = self._report(fetch=True, refresh=refresh)
        if not report.to_fetch:
            return
        with Progress(transient=True, console=Console(stderr=True)) as progress:
            task = progress.add_task('Downloading the data', total=len(report.to_fetch))
            for item in report.to_fetch:
                store.fetch([item], self._authorize)
                progress.advance(task)
        self._downloaded = None

    def _unprepared(self: Self) -> PreparationReport | None:
        """Return what a download would have to do, when that can be known without downloading.

        It cannot when the store does not hold the catalogue either, and finding out would mean downloading. Saying less
        is better than downloading in order to say more, since downloading is the thing that was not asked for.
        """
        try:
            return self._report(fetch=False)
        except KeyError:
            return None

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long `stats`/`odds` snapshots of the sources, read from the store.

        It never fetches. When the store is not prepared it fails loudly, so a data request can never download by
        surprise.
        """
        if self._downloaded is None:
            stats_source, odds_source, store = self._resolved()
            try:
                stats_items, odds_items = self._items(fetch=False)
                items = self._unique(stats_items + odds_items)
                payloads = {(payload.item.source, payload.item.key): payload for payload in store.read(items)}
            except KeyError as error:
                raise NotPreparedError(self._unprepared()) from error
            stats = self._derive(stats_source, [payloads[item.source, item.key] for item in stats_items], store)
            if odds_source is None:
                self._downloaded = (stats, self.no_odds())
                return self._downloaded
            odds = self._derive(odds_source, [payloads[item.source, item.key] for item in odds_items], store)
            if stats_source.name != odds_source.name and not odds.empty:
                aliases = {**ALIASES, **(self.aliases or {})}
                odds, self.reconciliation_ = resolve(stats, odds, aliases, self.max_unmatched_rate)
            self._downloaded = (stats, odds)
        return self._downloaded

    @staticmethod
    def _derive(source: BaseSource, payloads: list[RawPayload], store: BaseStore) -> pd.DataFrame:
        """Return the snapshots of a source, kept so they are not rebuilt on every extraction.

        They are a cache, not an archive: they are rebuilt from the raw payloads at no cost, so changing the transform
        never costs anything.

        This is the seam every source's snapshots pass through, so it is where their instants are normalized to UTC. A
        source that emitted a naive instant could not be compared with one that did not.
        """
        if not isinstance(store, LocalStore):
            return BaseDataLoader._finalize(source.to_snapshots(payloads))
        digest = payloads_digest(payloads, source.transform_digest())
        snapshots = store.read_snapshots(source.name, source.kind, digest)
        if snapshots is None:
            snapshots = BaseDataLoader._finalize(source.to_snapshots(payloads))
            store.write_snapshots(source.name, source.kind, digest, snapshots)
        return snapshots
