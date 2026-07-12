"""Download and transform historical and fixtures basketball data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar

from .._base._sourced import SourcedDataLoader
from .._sources._base import BaseOddsSource, BaseStatsSource
from .._sources._euroleague import EuroLeagueStats


class BasketballDataLoader(SourcedDataLoader):
    """Dataloader for basketball data.

    It reads long event-snapshot `stats` and `odds` data from the injected sources, and derives the providers, markets,
    per-column metadata and moment-aware training and fixtures data from the data itself, exactly as the soccer
    dataloader does. Nothing in the engine knows which sport it is looking at.

    Basketball cannot be drawn, since a tie goes to overtime, so the outcome is two-way. That is not configured
    anywhere: the markets are read from the columns the data carries, so a dataset with no draw simply has none.

    **There is no free source of basketball odds**, so `odds` has no default and must be given. A dataloader without one
    carries no betting markets, and therefore has nothing to predict.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.

        stats:
            The source of the statistics. The default `None` uses the free
            official API of the EuroLeague.

        odds:
            The source of the odds. It has no default, since no basketball odds
            are free: pass your own, e.g. `OddsApi(key=...)`.

        store:
            Where the downloaded data is kept. The default `None` keeps it in
            `~/.sportsbet`.

        aliases:
            The team names of the odds source, mapped to the names of the
            statistics source, for when the two do not name a club the same way.

        max_unmatched_rate:
            The proportion of games that may go without odds. The default `0.0`
            allows none. Basketball always mixes two sources, so this is the
            normal path rather than an edge case.

    Attributes:
        reconciliation_ (ReconciliationReport):
            How well the games of the two sources were found in each other.
    """

    DEFAULT_STATS: ClassVar[type[BaseStatsSource] | None] = EuroLeagueStats
    DEFAULT_ODDS: ClassVar[type[BaseOddsSource] | None] = None
