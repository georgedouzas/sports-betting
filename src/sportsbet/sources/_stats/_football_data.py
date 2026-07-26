"""Read the football-data.co.uk statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from .._base import BaseStatsSource
from .._common._football_data import _FootballDataSource


class FootballDataStats(_FootballDataSource, BaseStatsSource):
    """The soccer schedule, results and match statistics from the football-data.co.uk feed.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.dataloaders import DataLoader
        >>> from sportsbet.sources import FootballDataOdds, FootballDataStats
        >>> source = FootballDataStats()
        >>> source.name, source.kind, source.sport
        ('football_data', 'stats', 'soccer')
        >>> # It declares what it would read to learn what it publishes, and reads nothing.
        >>> [item.key for item in source.list_index_items({'league': ['Italy']})]
        ['index_Italy']
        >>> # Hand it to a dataloader, together with wherever the odds come from.
        >>> dataloader = DataLoader(
        ...     param_grid={'league': ['Italy'], 'division': [1], 'year': [2024]},
        ...     stats=source,
        ...     odds=FootballDataOdds(),
        ... )
        >>> dataloader.sport_
        'soccer'
    """
