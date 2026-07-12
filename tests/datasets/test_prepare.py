"""Tests for the preparation of the data."""

import pandas as pd
import pytest

from sportsbet.datasets import (
    BaseOddsSource,
    BaseStatsSource,
    LocalStore,
    NotPreparedError,
    RawItem,
    SoccerDataLoader,
)

PARAMS = [{'league': 'England', 'division': 1, 'year': 2024}]
SEASON = b'league,year\nEngland,2024\n'
PRICED_ITEMS = 2
FREE_ITEMS = 1
BOTH_SOURCES = 2


class _Feed:
    """A source declaring one free catalogue item and one item per season."""

    name = 'feed'
    cost = 0

    def index_items(self):
        return [RawItem(source=self.name, key='catalogue', url='https://example.com/catalogue.csv', volatile=True)]

    def catalogue(self, payloads):
        return PARAMS

    def required_items(self, params):
        return [
            RawItem(
                source=self.name,
                key=f'{param["league"]}_{param["division"]}_{param["year"]}',
                url=f'https://example.com/{param["league"]}.csv',
                cost=self.cost,
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        return pd.DataFrame({'key': [payload.item.key for payload in payloads]})


class _FeedStats(_Feed, BaseStatsSource):
    """The statistics of the feed."""


class _FeedOdds(_Feed, BaseOddsSource):
    """The odds of the feed."""


class _PricedOdds(_FeedOdds):
    """An odds source that charges for its data."""

    name = 'priced'
    cost = 620


@pytest.fixture
def offline(monkeypatch):
    """Count the fetches and serve them without the network."""
    fetches = []

    def read_urls_content(urls):
        fetches.extend(urls)
        return [SEASON for _ in urls]

    monkeypatch.setattr('sportsbet.datasets._store.read_urls_content', read_urls_content)
    return fetches


@pytest.fixture
def loader(tmp_path):
    """A dataloader backed by a feed and a store in a temporary directory."""
    return SoccerDataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))


def test_extraction_without_preparation_fails(loader, offline):
    """Test extraction never fetches; it fails loudly instead, so nothing downloads by surprise."""
    with pytest.raises(NotPreparedError, match='Call `prepare` first'):
        loader.extract_train_data()


def test_the_failure_says_what_is_missing(loader, offline):
    """Test the failure names what is missing, rather than leaving the user to guess."""
    with pytest.raises(NotPreparedError) as failure:
        loader.extract_train_data()
    assert [item.key for item in failure.value.report.to_fetch] == ['England_1_2024']


def test_a_dry_run_fetches_no_data(loader, offline):
    """Test a dry run says what a preparation would do without fetching the data."""
    report = loader.prepare(dry_run=True)
    assert [item.key for item in report.to_fetch] == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_a_dry_run_prices_the_data_without_paying(tmp_path, offline):
    """Test the cost of a preparation is known before it is paid."""
    loader = SoccerDataLoader(stats=_FeedStats(), odds=_PricedOdds(), store=LocalStore(tmp_path))
    report = loader.prepare(dry_run=True)
    assert report.estimated_cost == {'priced': 620}
    assert not [url for url in offline if url.endswith('England.csv')]


def test_preparing_fetches_the_data(loader, offline):
    """Test a preparation fetches what the store does not hold."""
    report = loader.prepare()
    assert [item.key for item in report.to_fetch] == ['England_1_2024']
    assert 'https://example.com/England.csv' in offline


def test_preparing_again_fetches_nothing(loader, offline):
    """Test a preparation is incremental, so a completed one costs nothing to repeat."""
    loader.prepare()
    offline.clear()
    loader.prepare()
    assert not [url for url in offline if url.endswith('England.csv')]


def test_extraction_after_preparation_reads_the_store(loader, offline):
    """Test extraction reads what the preparation fetched, and fetches nothing itself."""
    loader.prepare()
    offline.clear()
    stats, odds = loader._snapshots()
    assert list(stats['key']) == ['England_1_2024']
    assert list(odds['key']) == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_rebuilding_the_snapshots_fetches_nothing(loader, offline, tmp_path):
    """Test the derived data is rebuilt from the raw payloads, so changing the transform never costs anything."""
    loader.prepare()
    for path in (tmp_path / 'snapshots').rglob('*.parquet'):
        path.unlink()
    offline.clear()
    rebuilt = SoccerDataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))
    stats, _ = rebuilt._snapshots()
    assert list(stats['key']) == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_an_item_two_sources_share_is_fetched_once(loader, offline):
    """Test a file the statistics and the odds share is downloaded once and not twice."""
    loader.prepare()
    assert offline.count('https://example.com/England.csv') == 1


def test_unavailable_params_are_reported(tmp_path, offline):
    """Test a combination the source does not publish is reported at plan time, not as a fetch time failure."""
    loader = SoccerDataLoader(
        param_grid={'league': ['England'], 'division': [1], 'year': [1800]},
        stats=_FeedStats(),
        odds=_FeedOdds(),
        store=LocalStore(tmp_path),
    )
    report = loader.prepare(dry_run=True)
    assert report.unavailable == [{'league': 'England', 'division': 1, 'year': 1800}]


def test_a_finished_season_is_not_fetched_again(loader, offline):
    """Test a season that cannot change upstream is held, so it is never downloaded twice."""
    loader.prepare()
    offline.clear()
    loader.prepare()
    assert not [url for url in offline if url.endswith('England.csv')]


def test_refresh_fetches_what_is_held(loader, offline):
    """Test a correction upstream is picked up on request, since nothing published is truly immutable."""
    loader.prepare()
    offline.clear()
    report = loader.prepare(refresh=True)
    assert [item.key for item in report.to_fetch] == ['England_1_2024']
    assert 'https://example.com/England.csv' in offline


def test_a_changed_transform_rebuilds_the_derived_data(loader, offline, tmp_path, monkeypatch):
    """Test a change to the transform never serves what the previous one produced.

    The snapshots are derived by code as well as from data, so the code is part of their identity.
    """
    builds = []
    to_snapshots = _Feed.to_snapshots
    monkeypatch.setattr(_Feed, 'to_snapshots', lambda self, payloads: builds.append(1) or to_snapshots(self, payloads))

    loader.prepare()
    loader._snapshots()
    assert len(builds) == BOTH_SOURCES

    SoccerDataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))._snapshots()
    assert len(builds) == BOTH_SOURCES

    monkeypatch.setattr(_Feed, 'transform_digest', lambda self: 'changed', raising=False)
    SoccerDataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))._snapshots()
    assert len(builds) == BOTH_SOURCES * 2
