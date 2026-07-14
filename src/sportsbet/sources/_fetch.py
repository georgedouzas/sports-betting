"""Implements the asynchronous fetch layer shared by the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import aiohttp
import pandas as pd

CONNECTIONS_LIMIT = 20
ENCODING = 'ISO-8859-1'
LOCAL = 'file://'


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


def _read_local_content(url: str) -> bytes:
    """Read the content a `file://` URL points at."""
    return Path(url2pathname(urlparse(url).path)).read_bytes()


def read_urls_content(urls: list[str]) -> list[bytes]:
    """Read the URLs content.

    A `file://` URL is read from disk rather than over the network, so a source whose feed is a file that ships with the
    library is an ordinary source: the store fetches it, keeps it and skips it next time, exactly as it does a feed that
    lives on the internet.
    """
    remote = [url for url in urls if not url.startswith(LOCAL)]
    fetched = iter(asyncio.run(_read_urls_content_async(remote)) if remote else [])
    return [_read_local_content(url) if url.startswith(LOCAL) else next(fetched).encode(ENCODING) for url in urls]


def read_csv_content(content: bytes) -> pd.DataFrame:
    """Read a CSV from its raw content."""
    text = content.decode(ENCODING)
    names = pd.read_csv(io.StringIO(text), nrows=0, encoding=ENCODING).columns.to_list()
    return pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding=ENCODING, on_bad_lines='skip')
