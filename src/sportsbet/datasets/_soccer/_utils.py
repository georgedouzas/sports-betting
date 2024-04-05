"""Includes utilities for soccer data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io

import aiohttp
import pandas as pd

OVER_UNDER = [1.5, 2.5, 3.5, 4.5]
OUTPUTS = [
    (
        'output__home_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] > data['target__away_team__full_time_goals'],
    ),
    (
        'output__away_win__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] < data['target__away_team__full_time_goals'],
    ),
    (
        'output__draw__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] == data['target__away_team__full_time_goals'],
    ),
    (
        f'output__over_{OVER_UNDER[0]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[0],
    ),
    (
        f'output__over_{OVER_UNDER[1]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[1],
    ),
    (
        f'output__over_{OVER_UNDER[2]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[2],
    ),
    (
        f'output__over_{OVER_UNDER[3]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        > OVER_UNDER[3],
    ),
    (
        f'output__under_{OVER_UNDER[0]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[0],
    ),
    (
        f'output__under_{OVER_UNDER[1]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[1],
    ),
    (
        f'output__under_{OVER_UNDER[2]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[2],
    ),
    (
        f'output__under_{OVER_UNDER[3]}__full_time_goals',
        lambda data: data['target__home_team__full_time_goals'] + data['target__away_team__full_time_goals']
        < OVER_UNDER[3],
    ),
]
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


def _read_csv(url: str) -> pd.DataFrame:
    """Read the CSV."""
    return _read_csvs([url])[0]
