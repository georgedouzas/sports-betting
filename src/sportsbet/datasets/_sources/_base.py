"""Implements the base classes of the data sources."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import pandas as pd

if TYPE_CHECKING:
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
    def index_items(self: Self) -> list[RawItem]:
        """Return the items needed to discover what the source publishes.

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

    def available_params(self: Self, store: BaseStore | None = None) -> list[dict]:
        """Return the parameter combinations the source publishes.

        This is where you start: a `param_grid` cannot be written before it is known what exists, so it needs no
        dataloader. The catalogue is free and it is kept in the store, so it is downloaded once.

        It depends on how the source is configured, since a credential may only cover part of what the source offers.

        Args:
            store:
                Where the catalogue is kept. The default `None` keeps it in `~/.sportsbet`.

        Returns:
            params:
                The available `league`, `division` and `year` combinations.
        """
        from .._store import LocalStore  # noqa: PLC0415

        store = store if store is not None else LocalStore()
        items = self.index_items()
        held = store.held(items)
        store.fetch([item for item in items if item not in held])
        return self.catalogue(store.read(items))

    @abstractmethod
    def required_items(self: Self, params: list[dict]) -> list[RawItem]:
        """Return the raw items the selected parameters need.

        Args:
            params:
                The selected parameter combinations.

        Returns:
            items:
                The items to fetch. Deterministic for the same parameters.
        """

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
