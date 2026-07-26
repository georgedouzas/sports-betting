"""Read the football-data.co.uk odds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from .._base import BaseOddsSource
from .._common._football_data import _FootballDataSource


class FootballDataOdds(_FootballDataSource, BaseOddsSource):
    """The pre-match closing odds of the market average and market maximum from the football-data.co.uk feed.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.sources import FootballDataOdds
        >>> source = FootballDataOdds()
        >>> source.name, source.kind, source.sport
        ('football_data', 'odds', 'soccer')
        >>> # It reads the same upstream files as the statistics.
        >>> from sportsbet.sources import FootballDataStats
        >>> stats_items = FootballDataStats().list_index_items({'league': ['Italy']})
        >>> source.list_index_items({'league': ['Italy']}) == stats_items
        True
    """
