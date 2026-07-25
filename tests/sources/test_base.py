"""Tests for the base of the data sources."""

import pandas as pd
import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import (
    BaseOddsSource,
    BaseSource,
    BaseStatsSource,
    RawItem,
    RawPayload,
    fetch_payloads,
    read_csv_content,
)
from tests.conftest import SnapshotsDataLoader

PARAMS = [{'league': 'England', 'division': 1, 'year': 2024}, {'league': 'Spain', 'division': 1, 'year': 2024}]
ITEM_COST = 2
TOTAL_COST = 4
DISTINCT_ITEMS = 2


class _TestSource(BaseSource):
    """A concrete test source whose items and snapshots are supplied directly."""

    name = 'test'

    def index_items(self, selection=None):
        return [RawItem(source=self.name, key='index', url='index.html')]

    def catalogue(self, payloads):
        return PARAMS

    def required_items(self, params, schedule=None):
        return [
            RawItem(
                source=self.name,
                key=f'{param["league"]}_{param["year"]}',
                url=f'{param["league"]}.csv',
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        return pd.concat([pd.read_csv(payload.content) for payload in payloads], ignore_index=True)


class _TestStatsSource(_TestSource, BaseStatsSource):
    """A concrete test statistics source."""


class _TestOddsSource(_TestSource, BaseOddsSource):
    """A concrete test odds source."""


def test_required_items_is_deterministic():
    """Test the same parameters declare the same items in the same order."""
    source = _TestSource()
    assert source.required_items(PARAMS) == source.required_items(PARAMS)


def test_items_are_identified_by_source_and_key():
    """Test two sources declaring the same key declare the same item, so it is fetched once."""
    item = RawItem(source='football_data', key='England_1_2024', url='a.csv')
    same = RawItem(source='football_data', key='England_1_2024', url='a.csv')
    other = RawItem(source='football_data', key='Spain_1_2024', url='b.csv')
    assert len({item, same, other}) == DISTINCT_ITEMS


def test_source_cannot_be_instantiated_without_the_contract():
    """Test a source must declare its items and transform its payloads."""
    cls = BaseSource
    with pytest.raises(TypeError, match='abstract'):
        cls()


def test_stats_and_odds_sources_are_distinguishable():
    """Test the dataloader can tell a statistics source from an odds source."""
    assert isinstance(_TestStatsSource(), BaseStatsSource)
    assert not isinstance(_TestOddsSource(), BaseStatsSource)


def test_payload_keeps_the_content_verbatim():
    """Test a payload is what the source returned, not a parse of it."""
    item = RawItem(source='test', key='a', url='a.csv')
    payload = RawPayload(item=item, content=b'league,year\nEngland,2024\n')
    assert payload.content == b'league,year\nEngland,2024\n'
    assert payload.item is item


def test_sources_are_stored_unmodified():
    """Test the sources are constructor parameters, kept as given."""
    stats, odds = _TestSource(), _TestSource()
    loader = DataLoader(stats=stats, odds=odds)
    assert loader.stats is stats
    assert loader.odds is odds


def test_a_dataloader_with_its_own_data_has_no_sources(stats, odds):
    """Test a dataloader carrying its own data reports no sources, since it downloads nothing."""
    assert SnapshotsDataLoader(stats, odds).sources == ()


def test_read_csv_content_reads_a_dataframe_from_bytes():
    """Test reading CSV content returns a data frame of its rows."""
    frame = read_csv_content(b'home,away\nArsenal,Chelsea\nEverton,Fulham\n')
    assert list(frame.columns) == ['home', 'away']
    assert frame.shape == (2, 2)
    assert frame['home'].tolist() == ['Arsenal', 'Everton']


def test_fetch_payloads_reads_each_item_and_keeps_the_order(tmp_path):
    """Test fetching the payloads reads each item from disk and pairs the bytes with it, in order."""
    files = []
    for name in ('a', 'b', 'c'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files.append(path)
    items = [RawItem(source='s', key=path.name, url=path.as_uri()) for path in files]
    payloads = fetch_payloads(items, lambda item: item.url)
    assert [payload.content for payload in payloads] == [b'a', b'b', b'c']
    assert [payload.item for payload in payloads] == items


def test_fetch_payloads_reads_the_url_from_authorize(tmp_path):
    """Test fetching the payloads reads from the URL authorize gives rather than the one on the item."""
    path = tmp_path / 'real'
    path.write_bytes(b'content')
    item = RawItem(source='s', key='k', url='https://example.invalid/never')
    payloads = fetch_payloads([item], lambda _: path.as_uri())
    assert payloads[0].content == b'content'
