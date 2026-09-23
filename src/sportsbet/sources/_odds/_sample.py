"""Read the sample soccer odds that ship with the library."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import ClassVar

from .._base import BaseOddsSource
from .._common._sample import SampleSource


class SampleSoccerOdds(SampleSource, BaseOddsSource):
    """The market average and market maximum pre-match odds of the bundled soccer sample season.

    Examples:
        >>> from sportsbet.sources import SampleSoccerOdds, fetch_payloads
        >>> source = SampleSoccerOdds()
        >>> source.name, source.kind, source.sport
        ('sample_soccer', 'odds', 'soccer')
        >>> # The bundled seasons are known up front, so nothing is read to learn them.
        >>> params = source.list_available_params()
        >>> items = source.list_required_items(params[:1])
        >>> [item.key for item in items]
        ['England_1_2024_odds']
        >>> # Reading the bundled file and shaping it gives the long snapshots.
        >>> snapshots = source.to_snapshots(fetch_payloads(items, source.request_url))
        >>> sorted(snapshots['provider'].unique())
        ['market_average', 'market_maximum']
    """

    kind: ClassVar[str] = 'odds'
