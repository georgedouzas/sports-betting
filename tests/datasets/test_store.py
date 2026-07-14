"""Tests for the store."""

import gzip
import json

import pandas as pd
import pytest

from sportsbet.sources import LocalStore, NotPreparedError, PreparationReport, RawItem

ITEM = RawItem(source='test', key='England_1_2024', url='https://example.com/England.csv')
VOLATILE = RawItem(source='test', key='fixtures', url='https://example.com/fixtures.csv', volatile=True)
CONTENT = b'league,year\nEngland,2024\n'


@pytest.fixture
def store(tmp_path):
    """A store kept in a temporary directory."""
    return LocalStore(tmp_path)


@pytest.fixture
def fetched(store, monkeypatch):
    """A store already holding the item, fetched without the network."""
    monkeypatch.setattr('sportsbet.sources._store.read_urls_content', lambda urls: [CONTENT for _ in urls])
    store.fetch([ITEM])
    return store


def test_nothing_is_held_by_an_empty_store(store):
    """Test an empty store holds nothing, so a preparation knows everything is missing."""
    assert store.held([ITEM]) == []


def test_a_fetched_item_is_held(fetched):
    """Test a fetched item is not fetched again."""
    assert fetched.held([ITEM]) == [ITEM]


def test_a_volatile_item_is_never_held(fetched, monkeypatch):
    """Test an item that can still change upstream is always refreshed, with no cache to invalidate by hand."""
    monkeypatch.setattr('sportsbet.sources._store.read_urls_content', lambda urls: [CONTENT for _ in urls])
    fetched.fetch([VOLATILE])
    assert fetched.held([VOLATILE]) == []


def test_the_raw_payload_is_kept_verbatim(fetched):
    """Test the raw payload is what came back, since metered data cannot be obtained again for free."""
    payload = fetched.read([ITEM])[0]
    assert payload.content == CONTENT
    assert payload.item == ITEM


def test_the_raw_payload_is_compressed(fetched, tmp_path):
    """Test the raw payload is kept compressed."""
    path = tmp_path / 'raw' / 'test' / 'England_1_2024.gz'
    assert gzip.decompress(path.read_bytes()) == CONTENT


def test_reading_an_item_that_is_not_held_fails(store):
    """Test the store never invents data it does not have."""
    with pytest.raises(KeyError, match='does not hold'):
        store.read([ITEM])


def test_the_manifest_records_what_is_held(fetched, tmp_path):
    """Test the index of the held items records the source, the key and a digest of the content."""
    entries = [json.loads(line) for line in (tmp_path / 'manifest.jsonl').read_text().splitlines()]
    assert entries[0]['source'] == 'test'
    assert entries[0]['key'] == 'England_1_2024'
    assert entries[0]['bytes'] == len(CONTENT)


def test_a_truncated_manifest_is_tolerated(fetched, tmp_path):
    """Test an interrupted write leaves the store readable rather than half corrupt."""
    path = tmp_path / 'manifest.jsonl'
    with path.open('a') as manifest_file:
        manifest_file.write('{"source": "test", "key": "trunc')
    assert fetched.held([ITEM]) == [ITEM]


def test_no_temporary_file_survives_a_write(fetched, tmp_path):
    """Test a partial write is never readable under its final name."""
    assert not list(tmp_path.rglob('*.tmp'))


def test_the_snapshots_round_trip_their_dtypes(store):
    """Test the column types survive the store, so an empty result cannot degrade the data it is combined with."""
    data = pd.DataFrame(
        {
            'league': ['England'],
            'division': [1],
            'year': [2024],
            'home_win': [1.8],
            'date': pd.to_datetime(['2024-08-10'], utc=True),
        },
    )
    store.write_snapshots('test', 'odds', 'digest', data)
    pd.testing.assert_frame_equal(store.read_snapshots('test', 'odds', 'digest'), data)


def test_snapshots_that_were_not_kept_are_absent(store):
    """Test the derived snapshots are a cache, so a miss is reported rather than invented."""
    assert store.read_snapshots('test', 'odds', 'missing') is None


def test_the_report_renders_what_it_would_do():
    """Test the report says what would be fetched and what it would cost."""
    report = PreparationReport(to_fetch=[ITEM], held=[], estimated_cost={'odds_api': 1240})
    assert 'Items to fetch: 1' in str(report)
    assert 'odds_api: 1240' in str(report)


def test_the_error_carries_the_report():
    """Test a failure to extract says exactly what is missing and what obtaining it would cost."""
    report = PreparationReport(to_fetch=[ITEM], estimated_cost={'odds_api': 1240})
    error = NotPreparedError(report)
    assert error.report is report
    assert 'Call `prepare` first' in str(error)
    assert 'odds_api: 1240' in str(error)
