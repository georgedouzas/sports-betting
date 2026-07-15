"""Tests for how extraction fetches, in memory and on demand."""

import pandas as pd
import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import BaseStatsSource, RawItem, RawPayload

PARAMS = [{'league': 'England', 'division': 1, 'year': 2024}]
CURRENT = {'league': 'England', 'division': 1, 'year': 2026}


def _rows(keys: list[str], *, played: bool) -> list[dict]:
    """Return one match per key, played in the past or upcoming."""
    date = pd.Timestamp('2024-08-16 19:00', tz='UTC') if played else pd.Timestamp.now(tz='UTC') + pd.Timedelta('7D')
    identity = {'league': 'England', 'division': 1, 'year': 2024 if played else 2026}
    rows = []
    for i in range(len(keys)):
        match = {'date': date, **identity, 'home_team': f'H{i}', 'away_team': f'A{i}'}
        rows.append({**match, 'event_status': 'preplay', 'event_time': 0, 'home_form': 1.0})
        if played:
            rows.append({**match, 'event_status': 'postplay', 'event_time': 0, 'home_win': 1})
    return rows


class _Feed:
    """A source recording which items it was asked to turn into snapshots."""

    def __init__(self: '_Feed', fetched: list[str]) -> None:
        self.fetched = fetched

    def index_items(self, selection=None):
        return [RawItem(source=self.name, key='catalogue', url='file:///dev/null')]

    def catalogue(self, payloads):
        return [*PARAMS, CURRENT]

    def required_items(self, params, schedule=None):
        return [RawItem(source=self.name, key=f'train_{param["year"]}', url='file:///dev/null') for param in params]

    def fixtures_items(self, params, schedule=None):
        return [RawItem(source=self.name, key='current', url='file:///dev/null')]

    def to_snapshots(self, payloads):
        self.fetched.extend(payload.item.key for payload in payloads)
        keys = [payload.item.key for payload in payloads]
        return pd.DataFrame(_rows(keys, played=all('current' not in key and 'train' in key for key in keys)))


class _FeedStats(_Feed, BaseStatsSource):
    """The statistics of the recording feed."""

    name = 'feed'
    sport = 'soccer'


@pytest.fixture
def fetched() -> list[str]:
    """The keys each extraction turned into snapshots."""
    return []


def _fetch_payloads(items, authorize):
    """Return a payload per item without reading anything."""
    return [RawPayload(item=item, content=b'') for item in items]


@pytest.fixture
def dataloader(fetched: list[str], monkeypatch: pytest.MonkeyPatch) -> DataLoader:
    """A dataloader over the recording feed, whose fetches are answered without the network."""
    monkeypatch.setattr('sportsbet.dataloaders._sourced.fetch_payloads', _fetch_payloads)
    return DataLoader(param_grid={'league': ['England'], 'division': [1], 'year': [2024]}, stats=_FeedStats(fetched))


def test_training_fetches_only_the_selected_seasons(dataloader: DataLoader, fetched: list[str]) -> None:
    """Test extracting the training data downloads the selected seasons and nothing of the fixtures."""
    dataloader.extract_train_data(learning_type='unsupervised')
    assert fetched == ['train_2024']


def test_fixtures_fetch_the_current_season_not_the_selected_one(dataloader: DataLoader, fetched: list[str]) -> None:
    """Test extracting the fixtures downloads the current data, separately from the training seasons."""
    dataloader.extract_train_data(learning_type='unsupervised')
    fetched.clear()
    dataloader.extract_fixtures_data()
    assert 'current' in fetched
    assert 'train_2024' not in fetched


def test_extracting_again_downloads_again(dataloader: DataLoader, fetched: list[str]) -> None:
    """Test each extraction downloads afresh, so the object always carries the latest data."""
    dataloader.extract_train_data(learning_type='unsupervised')
    fetched.clear()
    dataloader.extract_train_data(learning_type='unsupervised')
    assert 'train_2024' in fetched


@pytest.mark.parametrize('drop_na_thres', [1.5, -0.5])
def test_a_threshold_that_is_not_a_proportion_is_refused(
    dataloader: DataLoader,
    drop_na_thres: float,
) -> None:
    """Test `drop_na_thres` is a proportion, so a value outside `[0, 1]` is refused rather than misread."""
    with pytest.raises(ValueError, match='drop_na_thres'):
        dataloader.extract_train_data(learning_type='unsupervised', drop_na_thres=drop_na_thres)
