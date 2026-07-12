"""Tests for the odds source backed by The Odds API.

The vendor forbids redistributing its data and charges for every request, so the source is exercised against payloads
shaped like the documented responses. No key is used and no credit is spent.
"""

import json

import pandas as pd
import pytest

from sportsbet.datasets import OddsApi, RawPayload

SPORTS = json.dumps(
    [
        {'key': 'soccer_epl', 'group': 'Soccer', 'title': 'EPL', 'active': True},
        {'key': 'soccer_spain_la_liga', 'group': 'Soccer', 'title': 'La Liga', 'active': True},
        {'key': 'americanfootball_nfl', 'group': 'American Football', 'title': 'NFL', 'active': True},
    ],
).encode()
KICKOFF = pd.Timestamp('2024-08-16 19:00', tz='UTC')
FIRST_YEAR = 2021
BEFORE_HISTORY = 2019
HISTORICAL_MULTIPLIER = 10
MORE_MARKETS_AND_REGIONS = 4
SCHEDULE = pd.DataFrame(
    [
        {
            'date': KICKOFF,
            'league': 'England',
            'division': 1,
            'year': 2025,
            'home_team': 'Man United',
            'away_team': 'Fulham',
        },
        {
            'date': KICKOFF,
            'league': 'England',
            'division': 1,
            'year': 2025,
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
        },
    ],
)


def _event(home, away, kickoff=KICKOFF, last_update='2024-08-16T18:55:00Z'):
    """Build an event the way the vendor returns one."""
    return {
        'id': f'{home}{away}',
        'sport_key': 'soccer_epl',
        'commence_time': kickoff.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'home_team': home,
        'away_team': away,
        'bookmakers': [
            {
                'key': 'pinnacle',
                'title': 'Pinnacle',
                'last_update': last_update,
                'markets': [
                    {
                        'key': 'h2h',
                        'outcomes': [
                            {'name': home, 'price': 1.8},
                            {'name': away, 'price': 4.2},
                            {'name': 'Draw', 'price': 3.5},
                        ],
                    },
                    {
                        'key': 'totals',
                        'outcomes': [
                            {'name': 'Over', 'price': 1.9, 'point': 2.5},
                            {'name': 'Under', 'price': 1.95, 'point': 2.5},
                            {'name': 'Over', 'price': 2.6, 'point': 3.5},
                        ],
                    },
                ],
            },
        ],
    }


@pytest.fixture
def source():
    """An odds source with a key that is never used, since nothing is fetched."""
    return OddsApi(key='secret-key')


def _payload(source, key, content):
    """Build the payload of an item the source declared."""
    item = next(item for item in source.required_items([], SCHEDULE) if item.key == key)
    return RawPayload(item=item, content=content)


def test_the_catalogue_covers_only_the_mapped_leagues(source):
    """Test a competition the vendor lists but the library does not map is left out rather than guessed at."""
    payloads = [RawPayload(item=source.index_items()[0], content=SPORTS)]
    params = source.catalogue(payloads)
    leagues = {param['league'] for param in params}
    assert leagues == {'England', 'Spain'}


def test_the_catalogue_starts_where_the_history_does(source):
    """Test the vendor's history begins in 2020, so earlier seasons are never offered."""
    payloads = [RawPayload(item=source.index_items()[0], content=SPORTS)]
    years = {param['year'] for param in source.catalogue(payloads)}
    assert min(years) == FIRST_YEAR
    assert BEFORE_HISTORY not in years


def test_the_catalogue_is_free(source):
    """Test the catalogue costs nothing, so a preparation can be priced without spending anything."""
    assert source.estimate(source.index_items()) == 0


def test_matches_that_kick_off_together_share_a_snapshot(source):
    """Test one snapshot prices every match played at that instant, so it is paid for once."""
    items = source.required_items([], SCHEDULE)
    historical = [item for item in items if 'live' not in item.key]
    assert len(historical) == len(source._settings()[2])


def test_the_key_never_reaches_an_item(source):
    """Test the credential is never written to the store."""
    items = source.required_items([], SCHEDULE)
    assert not [item for item in items if 'secret-key' in item.url or 'apiKey' in item.url]
    assert 'apiKey=secret-key' in source.request_url(items[0])


def test_the_cost_follows_the_vendor(source):
    """Test a historical snapshot costs ten times a live one, per market and per region."""
    items = source.required_items([], SCHEDULE)
    historical = next(item for item in items if 'live' not in item.key)
    live = next(item for item in items if 'live' in item.key)
    markets, regions, _ = source._settings()
    assert historical.cost == HISTORICAL_MULTIPLIER * len(markets) * len(regions)
    assert live.cost == len(markets) * len(regions)


def test_more_markets_and_regions_cost_more():
    """Test every market and every region multiplies the cost, so the estimate reflects the settings."""
    cheap = OddsApi(key='k', markets=['h2h'], regions=['eu'])
    dear = OddsApi(key='k', markets=['h2h', 'totals'], regions=['eu', 'uk'])
    cheap_items = [item for item in cheap.required_items([], SCHEDULE) if 'live' not in item.key]
    dear_items = [item for item in dear.required_items([], SCHEDULE) if 'live' not in item.key]
    assert dear.estimate(dear_items) == MORE_MARKETS_AND_REGIONS * cheap.estimate(cheap_items)


def test_nothing_is_asked_for_without_a_schedule(source):
    """Test a season alone does not say when its matches are played, so nothing is requested."""
    assert source.required_items([{'league': 'England', 'division': 1, 'year': 2025}]) == []


def test_the_inplay_snapshot_carries_the_price_at_that_minute(source):
    """Test the odds are those available at the minute the bet would be placed, which is the point of the source."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if item.key.endswith('inplay__45'))
    content = json.dumps(
        {
            'timestamp': '2024-08-16T19:45:00Z',
            'data': [_event('Man United', 'Fulham'), _event('Arsenal', 'Chelsea')],
        },
    ).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    assert set(snapshots['event_status']) == {'inplay'}
    assert set(snapshots['event_time']) == {45}
    assert set(snapshots['provider']) == {'pinnacle'}
    assert snapshots.set_index('home_team').loc['Man United', 'home_win'] == pytest.approx(1.8)


def test_the_markets_are_named_the_way_the_library_names_them(source):
    """Test the vendor's head-to-head and totals become the library's markets."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if item.key.endswith('preplay__0'))
    content = json.dumps({'timestamp': '2024-08-16T18:59:00Z', 'data': [_event('Arsenal', 'Chelsea')]}).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    row = snapshots.iloc[0]
    assert row['home_win'] == pytest.approx(1.8)
    assert row['away_win'] == pytest.approx(4.2)
    assert row['draw'] == pytest.approx(3.5)
    assert row['over_2.5'] == pytest.approx(1.9)
    assert row['under_2.5'] == pytest.approx(1.95)


def test_a_match_the_snapshot_was_not_asked_for_is_left_out(source):
    """Test a snapshot prices every match running at that instant, but only the ones it was asked for are kept."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if item.key.endswith('inplay__45'))
    other = _event('Everton', 'Spurs', kickoff=pd.Timestamp('2024-08-16 14:00', tz='UTC'))
    content = json.dumps({'timestamp': '2024-08-16T19:45:00Z', 'data': [_event('Arsenal', 'Chelsea'), other]}).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    assert set(snapshots['home_team']) == {'Arsenal'}


def test_the_season_comes_from_the_statistics(source):
    """Test the season is the one the statistics gave, not one guessed from the kick-off month."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if item.key.endswith('preplay__0'))
    content = json.dumps({'timestamp': '2024-08-16T18:59:00Z', 'data': [_event('Arsenal', 'Chelsea')]}).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    assert set(snapshots['year']) == {2025}
    assert set(snapshots['league']) == {'England'}
    assert set(snapshots['division']) == {1}


def test_the_live_endpoint_prices_a_running_match_at_its_minute(source):
    """Test a match already under way is priced at the minute it has reached, so a live bet can be served."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if 'live' in item.key)
    running = _event('Arsenal', 'Chelsea', last_update='2024-08-16T19:30:00Z')
    content = json.dumps([running]).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    assert set(snapshots['event_status']) == {'inplay'}
    assert set(snapshots['event_time']) == {30}


def test_the_live_endpoint_prices_an_upcoming_match_before_kickoff(source):
    """Test a match that has not started is priced pre-play, so a fixture can be served."""
    key = next(item.key for item in source.required_items([], SCHEDULE) if 'live' in item.key)
    upcoming = _event('Arsenal', 'Chelsea', last_update='2024-08-16T18:30:00Z')
    content = json.dumps([upcoming]).encode()
    snapshots = source.to_snapshots([_payload(source, key, content)])
    assert set(snapshots['event_status']) == {'preplay'}
    assert set(snapshots['event_time']) == {0}
