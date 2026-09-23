"""Read the football-data.co.uk statistics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BaseStatsSource
from .._common._football_data import FootballDataSource


class FootballDataStats(FootballDataSource, BaseStatsSource):
    """The soccer schedule, results and match statistics from the football-data.co.uk feed.

    Read more in the [user guide][user-guide].

    Examples:
        >>> from sportsbet.sources import FootballDataStats
        >>> source = FootballDataStats()
        >>> source.name, source.kind, source.sport
        ('football_data', 'stats', 'soccer')
        >>> # It declares what it would read to learn what it publishes, and reads nothing.
        >>> [item.key for item in source.list_index_items({'league': ['Italy']})]
        ['index_Italy']
        >>> # Asking for everything it publishes declares one index page per league.
        >>> len(source.list_index_items()) > 1
        True
    """
