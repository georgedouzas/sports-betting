"""Tests for the odds source backed by Lumify.

Exercised against payloads shaped like the documented responses. No key is used and no credit is spent.
"""

import json

import pandas as pd
import pytest

from sportsbet.sources import LumifyOdds, RawItem, RawPayload

KICKOFF = '2026-08-07T00:00:00Z'


def _event(home: str, away: str, *, status: str = 'scheduled') -> dict:
    """Build an event the way Lumify returns one with inlined odds."""
    return {
        'id': 101,
        'status': status,
        'starts_at': KICKOFF,
        'participants': [
            {'role': 'home', 'team': {'name': home}},
            {'role': 'away', 'team': {'name': away}},
        ],
        'odds': {
            'available': True,
            'bookmakers': [
                {
                    'bookmaker': 'pinnacle',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'outcome': home, 'price': -150},
                                {'outcome': away, 'price': 130},
                            ],
                        },
                        {
                            'key': 'totals',
                            'outcomes': [
                                {'outcome': 'Over', 'price': -110, 'point': 2.5},
                                {'outcome': 'Under', 'price': -110, 'point': 2.5},
                            ],
                        },
                    ],
                },
            ],
        },
    }


@pytest.fixture
def source(monkeypatch):
    """An odds source whose key is read from the environment, never fetched."""
    monkeypatch.setenv('LUMIFY_API_KEY', 'secret-key')
    return LumifyOdds(key_env='LUMIFY_API_KEY', markets=['h2h', 'totals'])


def test_the_catalogue_covers_mapped_leagues(source):
    """Test the fixed league map is what the catalogue publishes."""
    params = source.read_catalogue([])
    leagues = {param['league'] for param in params}
    assert {'NBA', 'England', 'Spain', 'USA'} <= leagues


def test_the_key_never_reaches_an_item(source):
    """Test the credential is never written to the store."""
    items = source.list_required_items([{'league': 'NBA', 'division': 1, 'year': 2026}])
    assert items
    assert not [item for item in items if 'secret-key' in item.url]
    assert 'Bearer secret-key' == source.request_headers(items[0])['Authorization']


def test_required_items_are_one_slate_per_league(source):
    """Test each selected league becomes one live-slate item."""
    items = source.list_required_items(
        [
            {'league': 'NBA', 'division': 1, 'year': 2026},
            {'league': 'NBA', 'division': 1, 'year': 2026},
            {'league': 'England', 'division': 1, 'year': 2026},
        ],
    )
    assert len(items) == 2
    assert {item.key for item in items} == {'nba__-__2026__live', 'soccer__epl__2026__live'}


def test_to_snapshots_converts_american_prices(source):
    """Test American moneylines become decimal market columns."""
    item = RawItem(source='lumify', key='nba__-__2026__live', url='https://lumify.ai/v1/events?sport=nba')
    payload = RawPayload(item=item, content=json.dumps([_event('Boston Celtics', 'LA Lakers')]).encode())
    snapshots = source.to_snapshots([payload])
    assert len(snapshots) == 1
    row = snapshots.iloc[0]
    assert row['provider'] == 'pinnacle'
    assert row['home_team'] == 'Boston Celtics'
    assert row['home_win'] == pytest.approx(1.6667, rel=1e-3)
    assert row['away_win'] == pytest.approx(2.3, rel=1e-3)
    assert row['over_2.5'] == pytest.approx(1.9091, rel=1e-3)
    assert row['event_status'] == 'preplay'
    assert pd.Timestamp(row['date']).tzinfo is not None
