"""Read the sample soccer statistics that ship with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import ClassVar

from .._base import BaseStatsSource
from .._common._sample import SampleSource


class SampleSoccerStats(SampleSource, BaseStatsSource):
    """A frozen real season of the English and Spanish soccer first divisions, bundled with the library.

    It carries the identity of every match, each team's form before it, the half-time score and the result. The season
    is finished.

    Examples:
        >>> from sportsbet.sources import SampleSoccerStats, fetch_payloads
        >>> source = SampleSoccerStats()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'stats', 'soccer')
        >>> # The bundled data is known up front, so nothing is read to learn it.
        >>> source.list_available_params()
        [{'division': 1, 'league': 'England', 'year': 2024}, {'division': 1, 'league': 'Spain', 'year': 2024}]
        >>> # Reading the bundled file and shaping it gives the long snapshots of one season.
        >>> items = source.list_required_items(source.list_available_params()[:1])
        >>> snapshots = source.to_snapshots(fetch_payloads(items, source.request_url))
        >>> sorted(snapshots['event_status'].unique())
        ['inplay', 'postplay', 'preplay']
    """

    kind: ClassVar[str] = 'stats'
