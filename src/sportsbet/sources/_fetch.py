"""Implements the asynchronous fetch layer shared by the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

import aiohttp
import pandas as pd

if TYPE_CHECKING:
    from ._base import RawItem, RawPayload

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
    """Read the content behind each URL.

    A `file://` URL is read from disk and the rest are read over the network, so a source whose feed ships with the
    library reads its files exactly as another source reads a remote feed.
    """
    remote = [url for url in urls if not url.startswith(LOCAL)]
    fetched = iter(asyncio.run(_read_urls_content_async(remote)) if remote else [])
    return [_read_local_content(url) if url.startswith(LOCAL) else next(fetched).encode(ENCODING) for url in urls]


def fetch_payloads(items: list[RawItem], authorize: Callable[[RawItem], str]) -> list[RawPayload]:
    """Read each item and pair the bytes back with the item that asked for it.

    Args:
        items:
            The items to read.

        authorize:
            Turns an item into the URL to read it from, adding a credential at the moment of the request so it stays out
            of the item.

    Returns:
        payloads:
            One payload per item, in the same order.
    """
    from ._base import RawPayload  # noqa: PLC0415

    contents = read_urls_content([authorize(item) for item in items])
    return [RawPayload(item=item, content=content) for item, content in zip(items, contents, strict=True)]


def read_csv_content(content: bytes) -> pd.DataFrame:
    """Read a CSV from its raw content."""
    text = content.decode(ENCODING)
    names = pd.read_csv(io.StringIO(text), nrows=0, encoding=ENCODING).columns.to_list()
    return pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding=ENCODING, on_bad_lines='skip')
