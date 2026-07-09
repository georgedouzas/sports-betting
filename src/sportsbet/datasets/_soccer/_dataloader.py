"""Download and transform historical and fixtures soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io
from typing import Self

import aiohttp
import pandas as pd

from .._base._dataloader import BaseDataLoader

BASE_URL = 'https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling'
STATS_URL = BASE_URL + '/stats/{league}_{division}_{year}.csv'
ODDS_URL = BASE_URL + '/odds/{league}_{division}_{year}.csv'
STATS_FIXTURES_URL = BASE_URL + '/stats/fixtures.csv'
ODDS_FIXTURES_URL = BASE_URL + '/odds/fixtures.csv'
PARAMS_URL = BASE_URL + '/params.csv'
CONNECTIONS_LIMIT = 20


async def _read_url_content_async(client: aiohttp.ClientSession, url: str) -> str:
    """Read asynchronously the URL content."""

    async with client.get(url) as response:
        with io.StringIO(await response.text(encoding='ISO-8859-1')) as text_io:
            return text_io.getvalue()


async def _read_urls_content_async(urls: list[str]) -> list[str]:
    """Read asynchronously the URLs content."""

    async with aiohttp.ClientSession(
        raise_for_status=True,
        connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
    ) as client:
        futures = [_read_url_content_async(client, url) for url in urls]
        return await asyncio.gather(*futures)


def _read_urls_content(urls: list[str]) -> list[str]:
    """Read the URLs content."""
    return asyncio.run(_read_urls_content_async(urls))


def _read_csvs(urls: list[str]) -> list[pd.DataFrame]:
    """Read the CSVs."""
    urls_content = _read_urls_content(urls)
    csvs = []
    for content in urls_content:
        names = pd.read_csv(io.StringIO(content), nrows=0, encoding='ISO-8859-1').columns.to_list()
        csv = pd.read_csv(io.StringIO(content), names=names, skiprows=1, encoding='ISO-8859-1', on_bad_lines='skip')
        csvs.append(csv)
    return csvs


class SoccerDataLoader(BaseDataLoader):
    """Dataloader for soccer data.

    It downloads long event-snapshot `stats` and `odds` data for the selected
    leagues, years and divisions, then derives the providers, markets, per-column
    metadata and moment-aware training and fixtures data from the data itself.
    Nothing about the feed is hardcoded: the available parameters come from the
    feed manifest, the moments come from the stored `event_status`/`event_time`,
    and each column's role is derived from where it actually carries values.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            Selects the data to include. Keys are parameters like `'league'`,
            `'division'` or `'year'` and values are sequences of allowed values,
            mirroring scikit-learn's `ParameterGrid`. The default `None` selects
            all available parameters.
    """

    @classmethod
    def _all_params(cls: type[Self]) -> list[dict]:
        """Return the available `league`/`division`/`year` combinations from the feed manifest."""
        manifest = _read_csvs([PARAMS_URL])[0]
        manifest = manifest[['league', 'division', 'year']].astype({'division': int, 'year': int})
        return manifest.to_dict('records')

    def _snapshots(self: Self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the long `stats`/`odds` snapshots (downloaded once, or user-provided)."""
        if self._provided_snapshots is not None:
            return self._provided_snapshots
        if self._downloaded is None:
            stats_urls = [STATS_URL.format(**params) for params in self._selected_params()]
            odds_urls = [ODDS_URL.format(**params) for params in self._selected_params()]
            stats = self._concat(_read_csvs([*stats_urls, STATS_FIXTURES_URL]))
            odds = self._concat(_read_csvs([*odds_urls, ODDS_FIXTURES_URL]))
            self._downloaded = (stats, odds)
        return self._downloaded
