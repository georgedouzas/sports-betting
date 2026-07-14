"""Implements the store that fetches and persists the data of the sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import gzip
import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

from ._fetch import read_urls_content

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._base import RawItem, RawPayload

    Authorize = Callable[[RawItem], str]

DEFAULT_PATH = Path.home() / '.sportsbet'


@dataclass
class PreparationReport:
    """What a download would do, or did.

    It says how many requests each source needs. It does not say what they cost: a vendor sets its own prices, changes
    them, and prices its endpoints differently, and a library that guessed at that would be quoting you a number it had
    made up. The requests are a fact. What they are worth is between you and whoever you buy them from.

    Attributes:
        to_fetch:
            The items missing from the store, or held but able to change upstream.

        held:
            The items the store already holds and that cannot change upstream.

        unavailable:
            The requested parameters the sources do not publish, reported instead of failing at fetch time.
    """

    to_fetch: list[RawItem] = field(default_factory=list)
    held: list[RawItem] = field(default_factory=list)
    unavailable: list[dict] = field(default_factory=list)

    @property
    def requests(self: Self) -> dict[str, int]:
        """The requests each source would have to make."""
        counted: dict[str, int] = {}
        for item in self.to_fetch:
            counted[item.source] = counted.get(item.source, 0) + 1
        return counted

    def __str__(self: Self) -> str:
        """Render the report."""
        requested = ', '.join(f'{source} {count:,}' for source, count in sorted(self.requests.items()))
        lines = [f'Requests to make: {requested}.' if requested else 'Nothing to download.']
        lines.append(f'Items held: {len(self.held)}.')
        if self.unavailable:
            lines.append(f'Unavailable parameters: {self.unavailable}.')
        return ' '.join(lines)


class NotPreparedError(Exception):
    """The store does not hold the data the selected parameters require.

    Nothing is downloaded unless it was asked for, so an extraction that has not been given `download=True` raises this
    instead, and says how many requests getting it would take.

    Args:
        report:
            What a download would have to do, when that can be known without downloading. It cannot when the store does
            not even hold the catalogue, and finding out would mean downloading.
    """

    def __init__(self: Self, report: PreparationReport | None = None) -> None:
        self.report = report
        msg = 'The data has not been downloaded. Pass `download=True` to get it.'
        super().__init__(f'{msg} {report}' if report is not None else msg)


class BaseStore(ABC):
    """The abstract base class for stores.

    A store fetches the raw items a source declares and persists them. It is the only place data is downloaded, so a
    source can never fetch by accident.
    """

    @abstractmethod
    def held(self: Self, items: list[RawItem]) -> list[RawItem]:
        """Return the items already held and unable to change upstream.

        Args:
            items:
                The items to check.

        Returns:
            held:
                The subset of the items the store holds.
        """

    @abstractmethod
    def fetch(self: Self, items: list[RawItem], authorize: Authorize | None = None) -> list[RawPayload]:
        """Fetch the items and persist their raw payloads.

        Args:
            items:
                The items to fetch.

            authorize:
                Returns the URL to fetch an item from. A source uses it to add its credential at the moment of the
                request, so the credential never reaches the store.

        Returns:
            payloads:
                The fetched payloads.
        """

    @abstractmethod
    def read(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Return the raw payloads of the items already held.

        Args:
            items:
                The items to read.

        Returns:
            payloads:
                The held payloads.
        """


def _write_atomic(path: Path, content: bytes) -> None:
    """Write the content, so a partial write is never readable under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as temporary_file:
        temporary_file.write(content)
    os.replace(temporary, path)  # noqa: PTH105


class LocalStore(BaseStore):
    """A store backed by the local filesystem.

    Raw payloads are kept forever, since metered data cannot be obtained again without paying for it again. Everything
    derived from them is rebuilt at no cost, so changing the transform never costs anything.

    Read more in the [user guide][user-guide].

    Args:
        path:
            Where to keep the data. The default `None` uses `~/.sportsbet`.
    """

    def __init__(self: Self, path: str | Path | None = None) -> None:
        self.path = path

    def _root(self: Self) -> Path:
        """Return the root of the store."""
        return Path(self.path) if self.path is not None else DEFAULT_PATH

    def _raw_path(self: Self, item: RawItem) -> Path:
        """Return where the raw payload of an item is kept."""
        return self._root() / 'raw' / item.source / f'{item.key}.gz'

    def _manifest_path(self: Self) -> Path:
        """Return where the index of the held items is kept."""
        return self._root() / 'manifest.jsonl'

    def _manifest(self: Self) -> dict[tuple[str, str], dict]:
        """Read the index of the held items, tolerating a truncated final line."""
        path = self._manifest_path()
        if not path.exists():
            return {}
        entries = {}
        with path.open() as manifest_file:
            for line in manifest_file:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries[entry['source'], entry['key']] = entry
        return entries

    def _record(self: Self, item: RawItem, content: bytes) -> None:
        """Append an item to the index of the held items, unless it says what the index already says.

        A volatile item is read again on every preparation, and a feed that has not changed answers with what it
        answered before. Writing that down again would grow the index for as long as the store is used, saying the same
        thing every time.
        """
        entry = {
            'source': item.source,
            'key': item.key,
            'volatile': item.volatile,
            'digest': hashlib.sha256(content).hexdigest(),
            'bytes': len(content),
        }
        if self._manifest().get((item.source, item.key)) == entry:
            return
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as manifest_file:
            manifest_file.write(json.dumps(entry) + '\n')

    def held(self: Self, items: list[RawItem]) -> list[RawItem]:
        """Return the items already held and unable to change upstream.

        A volatile item is never held, so it is always refreshed. There is no cache to invalidate by hand.

        Args:
            items:
                The items to check.

        Returns:
            held:
                The subset of the items the store holds.
        """
        manifest = self._manifest()
        return [
            item
            for item in items
            if not item.volatile and (item.source, item.key) in manifest and self._raw_path(item).exists()
        ]

    def fetch(self: Self, items: list[RawItem], authorize: Authorize | None = None) -> list[RawPayload]:
        """Fetch the items and persist their raw payloads.

        Args:
            items:
                The items to fetch.

            authorize:
                Returns the URL to fetch an item from. A source uses it to add its credential at the moment of the
                request, so the credential never reaches the store.

        Returns:
            payloads:
                The fetched payloads.
        """
        from ._base import RawPayload  # noqa: PLC0415

        if not items:
            return []
        urls = [authorize(item) if authorize is not None else item.url for item in items]
        contents = read_urls_content(urls)
        payloads = []
        for item, content in zip(items, contents, strict=True):
            _write_atomic(self._raw_path(item), gzip.compress(content))
            self._record(item, content)
            payloads.append(RawPayload(item=item, content=content))
        return payloads

    def read(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Return the raw payloads of the items already held.

        Args:
            items:
                The items to read.

        Returns:
            payloads:
                The held payloads.
        """
        from ._base import RawPayload  # noqa: PLC0415

        payloads = []
        for item in items:
            path = self._raw_path(item)
            if not path.exists():
                msg = f'The store does not hold the item `{item.key}` of the source `{item.source}`.'
                raise KeyError(msg)
            payloads.append(RawPayload(item=item, content=gzip.decompress(path.read_bytes())))
        return payloads

    def _snapshots_path(self: Self, source: str, kind: str, digest: str) -> Path:
        """Return where the snapshots derived from a set of payloads are kept."""
        return self._root() / 'snapshots' / source / kind / f'{digest}.parquet'

    def read_snapshots(self: Self, source: str, kind: str, digest: str) -> pd.DataFrame | None:
        """Return the snapshots derived from a set of payloads, if they were kept.

        Args:
            source:
                The name of the source.

            kind:
                Whether the snapshots are statistics or odds.

            digest:
                The identity of the payloads they were derived from.

        Returns:
            snapshots:
                The snapshots, or `None` if they were not kept.
        """
        path = self._snapshots_path(source, kind, digest)
        return pd.read_parquet(path) if path.exists() else None

    def write_snapshots(self: Self, source: str, kind: str, digest: str, data: pd.DataFrame) -> None:
        """Keep the snapshots derived from a set of payloads.

        They are a cache, not an archive: they are rebuilt from the raw payloads at no cost.

        Args:
            source:
                The name of the source.

            kind:
                Whether the snapshots are statistics or odds.

            digest:
                The identity of the payloads they were derived from.

            data:
                The snapshots to keep.
        """
        path = self._snapshots_path(source, kind, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        data.to_parquet(temporary, compression='zstd', index=False)
        os.replace(temporary, path)  # noqa: PTH105


def payloads_digest(payloads: list[RawPayload], transform: str) -> str:
    """Return the identity of a set of payloads and of the code that reads them.

    The transform is part of it because the snapshots are derived by code as well as from data. Without it, a change to
    the transform would leave the identity untouched and serve the snapshots the previous one produced.

    Args:
        payloads:
            The payloads the snapshots are derived from.

        transform:
            The identity of the code that turns them into snapshots.

    Returns:
        digest:
            The identity of the payloads and of the transform that reads them.
    """
    parts = sorted(f'{payload.item.key}:{hashlib.sha256(payload.content).hexdigest()}' for payload in payloads)
    parts.append(f'transform:{transform}')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:32]
