"""Tests for the preparation of the data."""

import contextlib

import pandas as pd
import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import (
    BaseOddsSource,
    BaseStatsSource,
    LocalStore,
    NotPreparedError,
    OddsApi,
    RawItem,
    SampleSoccerOdds,
    SampleSoccerStats,
)

PARAMS = [{'league': 'England', 'division': 1, 'year': 2024}]
SEASON = b'league,year\nEngland,2024\n'
PRICED_ITEMS = 2
FREE_ITEMS = 1
BOTH_SOURCES = 2


class _Feed:
    """A source declaring one free catalogue item and one item per season."""

    name = 'feed'

    def index_items(self, selection=None):
        return [RawItem(source=self.name, key='catalogue', url='https://example.com/catalogue.csv', volatile=True)]

    def catalogue(self, payloads):
        return PARAMS

    def required_items(self, params, schedule=None):
        return [
            RawItem(
                source=self.name,
                key=f'{param["league"]}_{param["division"]}_{param["year"]}',
                url=f'https://example.com/{param["league"]}.csv',
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        return pd.DataFrame(
            [
                {
                    'key': payload.item.key,
                    'event_status': 'preplay',
                    'event_time': 0,
                    'date': pd.Timestamp('2024-08-16 19:00'),
                    'league': 'England',
                    'division': 1,
                    'year': 2024,
                    'home_team': 'Arsenal',
                    'away_team': 'Chelsea',
                }
                for payload in payloads
            ],
        )


class _FeedStats(_Feed, BaseStatsSource):
    """The statistics of the feed."""


class _FeedOdds(_Feed, BaseOddsSource):
    """The odds of the feed."""


class _PricedOdds(_FeedOdds):
    """An odds source of its own, so it declares its own items."""

    name = 'priced'


SPORTS = b'[{"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": true}]'


@pytest.fixture
def offline(monkeypatch):
    """Count the fetches and serve them without the network."""
    fetches = []

    def read_urls_content(urls):
        fetches.extend(urls)
        return [SPORTS if 'the-odds-api' in url else SEASON for url in urls]

    monkeypatch.setattr('sportsbet.sources._store.read_urls_content', read_urls_content)
    return fetches


@pytest.fixture
def loader(tmp_path):
    """A dataloader backed by a feed and a store in a temporary directory."""
    return DataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))


def test_extraction_without_preparation_fails(loader, offline):
    """Test extraction never fetches; it fails loudly instead, so nothing downloads by surprise."""
    with pytest.raises(NotPreparedError, match='has not been downloaded'):
        loader.extract_train_data()


def test_the_failure_itself_never_fetches(loader, offline):
    """Test the failure does not download in order to describe itself.

    Saying what is missing means reading the catalogue, and reading the catalogue means downloading it. An extraction
    that downloads is the one thing this must never be, so it says less rather than fetching in order to say more.
    """
    with pytest.raises(NotPreparedError):
        loader.extract_train_data()
    assert not offline


def test_the_failure_says_what_is_missing_when_it_can(loader, offline):
    """Test the failure names what is missing, when it can know without downloading to find out."""
    stats_source, _ = loader.sources
    stats_source.available_params(store=loader.store)
    with pytest.raises(NotPreparedError) as failure:
        loader.extract_train_data()
    assert [item.key for item in failure.value.report.to_fetch] == ['England_1_2024']


def test_a_dry_run_fetches_no_data(loader, offline):
    """Test a dry run says what a preparation would do without fetching the data."""
    report = loader._report(fetch=True)
    assert [item.key for item in report.to_fetch] == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_preparing_fetches_the_data(loader, offline):
    """Test a preparation fetches what the store does not hold."""
    report = loader._report(fetch=True)
    loader._download(refresh=False)
    assert [item.key for item in report.to_fetch] == ['England_1_2024']
    assert 'https://example.com/England.csv' in offline


def test_preparing_again_fetches_nothing(loader, offline):
    """Test a preparation is incremental, so a completed one costs nothing to repeat."""
    loader._download(refresh=False)
    offline.clear()
    loader._download(refresh=False)
    assert not [url for url in offline if url.endswith('England.csv')]


def test_extraction_after_preparation_reads_the_store(loader, offline):
    """Test extraction reads what the preparation fetched, and fetches nothing itself."""
    loader._download(refresh=False)
    offline.clear()
    stats, odds = loader._snapshots()
    assert list(stats['key']) == ['England_1_2024']
    assert list(odds['key']) == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_rebuilding_the_snapshots_fetches_nothing(loader, offline, tmp_path):
    """Test the derived data is rebuilt from the raw payloads, so changing the transform never costs anything."""
    loader._download(refresh=False)
    for path in (tmp_path / 'snapshots').rglob('*.parquet'):
        path.unlink()
    offline.clear()
    rebuilt = DataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))
    stats, _ = rebuilt._snapshots()
    assert list(stats['key']) == ['England_1_2024']
    assert not [url for url in offline if url.endswith('England.csv')]


def test_an_item_two_sources_share_is_fetched_once(loader, offline):
    """Test a file the statistics and the odds share is downloaded once and not twice."""
    loader._download(refresh=False)
    assert offline.count('https://example.com/England.csv') == 1


def test_unavailable_params_are_reported(tmp_path, offline):
    """Test a combination the source does not publish is reported at plan time, not as a fetch time failure."""
    loader = DataLoader(
        param_grid={'league': ['England'], 'division': [1], 'year': [1800]},
        stats=_FeedStats(),
        odds=_FeedOdds(),
        store=LocalStore(tmp_path),
    )
    report = loader._report(fetch=True)
    assert report.unavailable == [{'league': 'England', 'division': 1, 'year': 1800}]


def test_a_finished_season_is_not_fetched_again(loader, offline):
    """Test a season that cannot change upstream is held, so it is never downloaded twice."""
    loader._download(refresh=False)
    offline.clear()
    loader._download(refresh=False)
    assert not [url for url in offline if url.endswith('England.csv')]


def test_refresh_fetches_what_is_held(loader, offline):
    """Test a correction upstream is picked up on request, since nothing published is truly immutable."""
    loader._download(refresh=False)
    offline.clear()
    report = loader._report(fetch=True, refresh=True)
    loader._download(refresh=True)
    assert 'England_1_2024' in [item.key for item in report.to_fetch]
    assert 'https://example.com/England.csv' in offline


def test_a_changed_transform_rebuilds_the_derived_data(loader, offline, tmp_path, monkeypatch):
    """Test a change to the transform never serves what the previous one produced.

    The snapshots are derived by code as well as from data, so the code is part of their identity.
    """
    builds = []
    to_snapshots = _Feed.to_snapshots
    monkeypatch.setattr(_Feed, 'to_snapshots', lambda self, payloads: builds.append(1) or to_snapshots(self, payloads))

    loader._download(refresh=False)
    loader._snapshots()
    assert len(builds) == BOTH_SOURCES

    DataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))._snapshots()
    assert len(builds) == BOTH_SOURCES

    monkeypatch.setattr(_Feed, 'transform_digest', lambda self: 'changed', raising=False)
    DataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))._snapshots()
    assert len(builds) == BOTH_SOURCES * 2


SNAPSHOTS = pd.DataFrame(
    [
        {
            'event_status': 'preplay',
            'event_time': 0,
            'date': pd.Timestamp('2024-08-16 19:00', tz='UTC'),
            'league': 'England',
            'division': 1,
            'year': 2025,
            'home_team': 'Man United',
            'away_team': 'Fulham',
        },
        {
            'event_status': 'preplay',
            'event_time': 0,
            'date': pd.Timestamp('2024-08-17 14:00', tz='UTC'),
            'league': 'England',
            'division': 1,
            'year': 2025,
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
        },
    ],
)


class _ScheduleStats(_FeedStats):
    """A free statistics source that says when the matches are played."""

    def to_snapshots(self, payloads):
        return SNAPSHOTS


def test_a_metered_odds_source_is_counted_exactly_without_being_asked(tmp_path, offline):
    """Test the requests a metered odds source needs are counted before one of them is made.

    The statistics are free and they say when the matches are played, so the snapshots the odds source needs can be
    counted exactly without asking it for any of them. Its catalogue is read, since that is free, but not one priced
    request is made. What those requests cost is between whoever is asking and the vendor.
    """
    odds = OddsApi(key='secret-key', markets=['h2h'], regions=['eu'])
    loader = DataLoader(stats=_ScheduleStats(), odds=odds, store=LocalStore(tmp_path))
    report = loader._report(fetch=True)

    snapshots = [item for item in report.to_fetch if item.source == 'odds_api']
    assert report.requests['odds_api'] == len(snapshots)
    assert not [url for url in offline if '/odds' in url]


def test_the_key_is_never_written_to_the_store(tmp_path, offline):
    """Test the credential reaches the request and nothing else."""
    odds = OddsApi(key='secret-key')
    loader = DataLoader(stats=_ScheduleStats(), odds=odds, store=LocalStore(tmp_path))
    report = loader._report(fetch=True)
    assert not [item for item in report.to_fetch if 'secret-key' in item.url]
    assert not [path for path in tmp_path.rglob('*') if path.is_file() and b'secret-key' in path.read_bytes()]


@pytest.mark.parametrize('drop_na_thres', [1.5, -0.5])
def test_a_threshold_that_is_not_a_proportion_is_refused(drop_na_thres):
    """Test a threshold outside nought and one says so.

    It is a proportion of the values that may be missing. Above one it asked for columns that are more than complete,
    and dropped every one of them instead of saying so; below nought it was accepted and meant nothing.
    """
    dataloader = DataLoader(param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
    with pytest.raises(ValueError, match='drop_na_thres'):
        dataloader.extract_train_data(drop_na_thres=drop_na_thres)


def test_a_preparation_reads_the_catalogue_once(loader, offline):
    """Test the index of a source is not read twice in the same preparation.

    The items and the parameters that are unavailable each used to resolve the catalogue for themselves, and the index
    is volatile and so is never held, so it was fetched twice and paid for twice. For a metered source that is the price
    of the catalogue, twice, on every preparation.
    """
    loader._download(refresh=False)
    catalogue = [url for url in offline if 'catalogue' in url]
    assert len(catalogue) == 1


def test_an_index_that_has_not_changed_is_not_written_down_again(loader, offline, tmp_path):
    """Test the index of the store does not grow for as long as the store is used.

    A volatile item is read again on every preparation, and a feed that has not changed answers with what it answered
    before. Writing that down every time would append a line per volatile item per preparation, forever.
    """
    loader._download(refresh=False)
    manifest = tmp_path / 'manifest.jsonl'
    lines = manifest.read_text().splitlines()

    DataLoader(stats=_FeedStats(), odds=_FeedOdds(), store=LocalStore(tmp_path))._download(refresh=False)
    assert manifest.read_text().splitlines() == lines


def test_an_extraction_downloads_only_when_it_is_asked_to(loader, offline):
    """Test nothing reaches the network unless a download was asked for.

    Downloading is the only thing that costs money, so it is the only thing that has to be asked for.
    """
    with pytest.raises(NotPreparedError, match='has not been downloaded'):
        loader.extract_train_data()
    assert not [url for url in offline if url.endswith('England.csv')]

    with contextlib.suppress(ValueError):
        loader.extract_train_data(download=True)
    assert [url for url in offline if url.endswith('England.csv')]


def test_the_report_counts_the_requests_it_would_make(loader, offline):
    """Test the report says how many requests each source would have to make, and makes none of them.

    It does not say what they cost. A vendor sets its own prices and changes them, and a library that guessed at that
    would be quoting a number it had made up.
    """
    report = loader._report(fetch=True)
    assert report.requests == {'feed': 1}
    assert not [url for url in offline if url.endswith('England.csv')]
