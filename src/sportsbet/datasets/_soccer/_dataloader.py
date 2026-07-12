"""Download and transform historical and fixtures soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from typing import ClassVar

from .._base._sourced import SourcedDataLoader
from .._sources._base import BaseOddsSource, BaseStatsSource
from .._sources._football_data import FootballDataOdds, FootballDataStats


class SoccerDataLoader(SourcedDataLoader):
    """Dataloader for soccer data.

    It reads long event-snapshot `stats` and `odds` data from the injected sources for the selected leagues, years and
    divisions, then derives the providers, markets, per-column metadata and moment-aware training and fixtures data
    from the data itself. Nothing about the sources is hardcoded: the available parameters come from the sources, the
    moments come from the stored `event_status`/`event_time`, and each column's role is derived from where it actually
    carries values.

    The data is downloaded by `prepare` and never by an extraction, so no data request can spend money or time by
    surprise. An extraction against a store that is not prepared fails loudly and says what is missing.

    Both sources default to the free football-data.co.uk feed, which is the only feed in the library carrying the
    statistics and the odds of a sport together.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.

        stats:
            The source of the statistics. The default `None` uses the free
            football-data.co.uk feed.

        odds:
            The source of the odds. The default `None` uses the free
            football-data.co.uk feed.

        store:
            Where the downloaded data is kept. The default `None` keeps it in
            `~/.sportsbet`.

        aliases:
            The team names of the odds source, mapped to the names of the
            statistics source, for when the two do not name a club the same way.

        max_unmatched_rate:
            The proportion of matches that may go without odds when the two
            sources are different. The default `0.0` allows none.

    Attributes:
        reconciliation_ (ReconciliationReport):
            How well the matches of the two sources were found in each other.
            Only set when they are different sources.
    """

    DEFAULT_STATS: ClassVar[type[BaseStatsSource] | None] = FootballDataStats
    DEFAULT_ODDS: ClassVar[type[BaseOddsSource] | None] = FootballDataOdds
