"""Implements the asynchronous fetch layer shared by the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io

import aiohttp
import pandas as pd

CONNECTIONS_LIMIT = 20
ENCODING = 'ISO-8859-1'


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


def read_urls_content(urls: list[str]) -> list[bytes]:
    """Read the URLs content."""
    contents = asyncio.run(_read_urls_content_async(urls))
    return [content.encode(ENCODING) for content in contents]


def read_csv_content(content: bytes) -> pd.DataFrame:
    """Read a CSV from its raw content."""
    text = content.decode(ENCODING)
    names = pd.read_csv(io.StringIO(text), nrows=0, encoding=ENCODING).columns.to_list()
    return pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding=ENCODING, on_bad_lines='skip')
