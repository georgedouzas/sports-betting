"""Implements the base classes of the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

if TYPE_CHECKING:
    from ... import ParamGrid
    from .._store import BaseStore


@dataclass(frozen=True)
class RawItem:
    """A unit of fetching, caching, resuming and cost.

    Two sources declaring the same `source` and `key` declare the same item, so data shared by a statistics and an odds
    source is fetched once rather than twice.

    Attributes:
        source:
            The name of the source that declared it.

        key:
            The identity of the item within the source.

        url:
            Where to fetch it from.

        volatile:
            Whether the content can still change upstream.

        cost:
            The units the source charges to fetch it.
    """

    source: str
    key: str
    url: str
    volatile: bool = False
    cost: int = 0


@dataclass(frozen=True)
class RawPayload:
    """What a source returned, kept verbatim.

    Attributes:
        item:
            What was asked for.

        content:
            Exactly what came back.
    """

    item: RawItem
    content: bytes


class BaseSource(ABC):
    """The abstract base class for data sources.

    A source declares the raw items a selection of parameters needs and turns the returned payloads into long snapshots.
    Its planning and transform methods never fetch, so a preparation can always be planned and priced without spending
    anything, and an extraction can never download by accident.

    It also answers what it publishes, through `available_params`. That question has to be answerable before a
    `param_grid` is written, so it belongs here and not on a dataloader that is configured with one.
    """

    name: ClassVar[str]
    kind: ClassVar[str]

    @abstractmethod
    def index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the items needed to discover what the source publishes.

        A feed that lists its seasons on an index page is cheap to ask. One that publishes a league as a single file of
        every season it ever played has no index, so the file is the catalogue, and reading it means downloading it.
        Asking such a feed what it publishes therefore costs as much as the data.

        So a source is told what is being looked for, and answers with what it needs in order to place it. A selection
        of one league has no business downloading another.

        Args:
            selection:
                What is being looked for. `None` asks for everything the source publishes, which is what discovery
                needs, since nothing can be selected before it is known what exists.

        Returns:
            items:
                The items of the catalogue. Always free, so a preparation can be priced without spending anything.
        """

    @abstractmethod
    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the parameter combinations the index payloads describe.

        Args:
            payloads:
                The payloads of the index items.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """

    def available_params(self: Self, store: BaseStore | None = None, refresh: bool = False) -> list[dict]:
        """Return the parameter combinations the source publishes.

        This is where you start: a `param_grid` cannot be written before it is known what exists, so it needs no
        dataloader. The catalogue is free and it is re-read, so a new season shows up as soon as it is published.

        What is published depends on how the source is configured, since a credential may only cover part of what the
        source offers.

        Args:
            store:
                Where the catalogue is kept. The default `None` keeps it in `~/.sportsbet`.

            refresh:
                If `True`, everything is read again, including what the store already holds.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """
        from .._store import LocalStore  # noqa: PLC0415

        store = store if store is not None else LocalStore()
        items = self.index_items()
        held = [] if refresh else store.held(items)
        store.fetch([item for item in items if item not in held], self.request_url)
        return self.catalogue(store.read(items))

    @abstractmethod
    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return the raw items the selected parameters need.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The matches of the selected parameters, with their kick-off instants. An odds source that addresses its
                prices by timestamp needs it, since a season alone does not say when its matches are played. It is
                `None` for a source that carries its own schedule.

        Returns:
            items:
                The items to fetch. Deterministic for the same parameters.
        """

    def needs_schedule(self: Self) -> bool:
        """Return whether the source has to be told when the matches are.

        A source that carries its own schedule, because its events and its odds arrive in the same file, does not.

        Returns:
            needed:
                Whether `required_items` needs a schedule.
        """
        return False

    def request_url(self: Self, item: RawItem) -> str:
        """Return the URL to fetch an item from.

        The credential is added here, at the moment of the request, so it never reaches a `RawItem` and is never
        written to the store.

        Args:
            item:
                The item to fetch.

        Returns:
            url:
                Where to fetch it from.
        """
        return item.url

    @abstractmethod
    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the raw payloads into a long snapshot table.

        Args:
            payloads:
                The payloads of the required items.

        Returns:
            snapshots:
                The long snapshots.
        """

    def transform_digest(self: Self) -> str:
        """Return the identity of the code that turns the payloads into snapshots.

        The snapshots are derived by code as well as from data, so what is cached against them has to know when the
        code changes. A release version cannot: an editable install keeps the version it was installed with, so an
        edited transform would keep serving what the previous one produced.

        Returns:
            digest:
                The identity of the transform.
        """
        try:
            module = Path(inspect.getfile(type(self)))
            return hashlib.sha256(module.read_bytes()).hexdigest()[:16]
        except (OSError, TypeError):
            return type(self).__name__

    def estimate(self: Self, items: list[RawItem]) -> int:
        """Return the units the source charges to fetch the items.

        Args:
            items:
                The items to fetch.

        Returns:
            cost:
                The estimated cost. Zero for free sources.
        """
        return sum(item.cost for item in items)


class BaseStatsSource(BaseSource):
    """The abstract base class for statistics sources."""

    kind: ClassVar[str] = 'stats'


class BaseOddsSource(BaseSource):
    """The abstract base class for odds sources."""

    kind: ClassVar[str] = 'odds'
