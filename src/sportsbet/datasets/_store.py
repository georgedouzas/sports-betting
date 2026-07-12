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
    from ._sources._base import RawItem, RawPayload

DEFAULT_PATH = Path.home() / '.sportsbet'


@dataclass
class PreparationReport:
    """The outcome of a preparation, or of what a preparation would do.

    Building it never spends anything, so the cost of a preparation is always known before it is paid.

    Attributes:
        to_fetch:
            The items missing from the store, or held but able to change upstream.

        held:
            The items the store already holds and that cannot change upstream.

        estimated_cost:
            The units each source would charge to fetch `to_fetch`. An estimate, not a measurement.

        unavailable:
            The requested parameters the sources do not publish, reported instead of failing at fetch time.
    """

    to_fetch: list[RawItem] = field(default_factory=list)
    held: list[RawItem] = field(default_factory=list)
    estimated_cost: dict[str, int] = field(default_factory=dict)
    unavailable: list[dict] = field(default_factory=list)

    def __str__(self: Self) -> str:
        """Render the report."""
        lines = [f'Items to fetch: {len(self.to_fetch)}.', f'Items held: {len(self.held)}.']
        if self.estimated_cost:
            costs = ', '.join(f'{source}: {cost}' for source, cost in self.estimated_cost.items())
            lines.append(f'Estimated cost: {costs}.')
        if self.unavailable:
            lines.append(f'Unavailable parameters: {self.unavailable}.')
        return ' '.join(lines)


class NotPreparedError(Exception):
    """The store does not hold the data the selected parameters require.

    Extraction never fetches, so it raises this instead. The report says what is missing and what obtaining it costs.

    Args:
        report:
            The preparation the store would need.
    """

    def __init__(self: Self, report: PreparationReport) -> None:
        self.report = report
        msg = f'The data is not prepared. Call `prepare` first. {report}'
        super().__init__(msg)


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
    def fetch(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Fetch the items and persist their raw payloads.

        Args:
            items:
                The items to fetch.

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
        """Append an item to the index of the held items."""
        entry = {
            'source': item.source,
            'key': item.key,
            'volatile': item.volatile,
            'digest': hashlib.sha256(content).hexdigest(),
            'bytes': len(content),
        }
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

    def fetch(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Fetch the items and persist their raw payloads.

        Args:
            items:
                The items to fetch.

        Returns:
            payloads:
                The fetched payloads.
        """
        from ._sources._base import RawPayload  # noqa: PLC0415

        if not items:
            return []
        contents = read_urls_content([item.url for item in items])
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
        from ._sources._base import RawPayload  # noqa: PLC0415

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


def payloads_digest(payloads: list[RawPayload]) -> str:
    """Return the identity of a set of payloads, so what is derived from them is cached against it.

    Args:
        payloads:
            The payloads the snapshots are derived from.

    Returns:
        digest:
            The identity of the payloads.
    """
    parts = sorted(f'{payload.item.key}:{hashlib.sha256(payload.content).hexdigest()}' for payload in payloads)
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:32]
