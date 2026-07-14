"""Tests for the statistics source backed by the EuroLeague's official API.

The API is fetched on the user's machine and nothing is redistributed, so the transform is exercised against a payload
shaped like the real response.
"""

import json

import pandas as pd
import pytest

from sportsbet.sources import EuroLeagueStats, RawPayload


def _game(code, utc, home, away, home_points, away_points, *, played):
    """Build a game the way the API returns one."""
    return {
        'gameCode': code,
        'utcDate': utc,
        'played': played,
        'local': {'club': {'name': home}, 'score': home_points},
        'road': {'club': {'name': away}, 'score': away_points},
    }


SNAPSHOTS = 7
HOME_POINTS = 77
AWAY_POINTS = 87
FORM_POINTS = 87
SEASONS = json.dumps({'data': [{'code': 'E2024', 'year': 2024}, {'code': 'E2023', 'year': 2023}]}).encode()
GAMES = json.dumps(
    {
        'data': [
            _game(1, '2024-10-03T16:45:00Z', 'Alba Berlin', 'Panathinaikos Athens', 77, 87, played=True),
            _game(2, '2024-10-03T16:30:00Z', 'Anadolu Efes Istanbul', 'Zalgiris Kaunas', 90, 80, played=True),
            _game(3, '2024-10-10T18:00:00Z', 'Panathinaikos Athens', 'Anadolu Efes Istanbul', 95, 85, played=True),
            _game(4, '2024-12-26T18:00:00Z', 'Alba Berlin', 'Zalgiris Kaunas', 0, 0, played=False),
        ],
    },
).encode()


@pytest.fixture
def source():
    """The statistics of the EuroLeague."""
    return EuroLeagueStats()


def _payloads(source):
    """Build the payload of a season the source declared."""
    item = source.required_items([{'league': 'Euroleague', 'division': 1, 'year': 2025}])[0]
    return [RawPayload(item=item, content=GAMES)]


def test_the_catalogue_comes_from_the_api(source):
    """Test the seasons are read from the competition, never fabricated from a range of years."""
    payloads = [RawPayload(item=source.index_items()[0], content=SEASONS)]
    assert source.catalogue(payloads) == [
        {'league': 'Euroleague', 'division': 1, 'year': 2024},
        {'league': 'Euroleague', 'division': 1, 'year': 2025},
    ]


def test_a_season_is_named_by_the_year_it_ends_in(source):
    """Test the API's `E2024` is the 2024-25 season, so the library calls it 2025."""
    payloads = [RawPayload(item=source.index_items()[0], content=SEASONS)]
    assert {params['year'] for params in source.catalogue(payloads)} == {2024, 2025}


def test_a_whole_season_is_one_request(source):
    """Test a season comes back in a single response, so it costs one request rather than one per round."""
    items = source.required_items([{'league': 'Euroleague', 'division': 1, 'year': 2025}])
    assert len(items) == 1
    assert 'seasons/E2024/games' in items[0].url


def test_the_tipoff_is_converted_from_the_time_of_the_competition(source):
    """Test the tip-off is the instant the API gives in UTC, not the one it gives in its own time.

    The API also publishes a time in its own head-office zone, whatever country the game is played in: a game in
    Istanbul reads 18:30 there and tips off at 20:30 locally. Reading that one would move every game by an hour or two,
    and nothing would say so.
    """
    snapshots = source.to_snapshots(_payloads(source))
    tipoffs = snapshots.set_index('home_team')['date']
    assert tipoffs['Anadolu Efes Istanbul'].iloc[0] == pd.Timestamp('2024-10-03 16:30')
    assert tipoffs['Alba Berlin'].iloc[0] == pd.Timestamp('2024-10-03 16:45')


def test_a_game_that_was_played_carries_its_result(source):
    """Test a played game becomes a post-play snapshot with its final score."""
    snapshots = source.to_snapshots(_payloads(source))
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert postplay.loc['Alba Berlin', 'home_points'] == HOME_POINTS
    assert postplay.loc['Alba Berlin', 'away_points'] == AWAY_POINTS


def test_a_game_that_was_not_played_is_only_a_fixture(source):
    """Test an unplayed game gets a pre-play snapshot and nothing else, so it never becomes a training row."""
    snapshots = source.to_snapshots(_payloads(source))
    assert len(snapshots) == SNAPSHOTS
    unplayed = snapshots[(snapshots['home_team'] == 'Alba Berlin') & (snapshots['away_team'] == 'Zalgiris Kaunas')]
    assert set(unplayed['event_status']) == {'preplay'}


def test_the_outcome_of_a_game_that_cannot_be_drawn(source):
    """Test basketball has no draw, since a tie goes to overtime, so the outcome is two-way."""
    snapshots = source.to_snapshots(_payloads(source))
    assert 'draw' not in snapshots.columns
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert postplay.loc['Alba Berlin', 'home_win'] == 0
    assert postplay.loc['Alba Berlin', 'away_win'] == 1
    assert postplay.loc['Anadolu Efes Istanbul', 'home_win'] == 1


def test_there_is_no_totals_market(source):
    """Test basketball has no totals line, so it has no totals market.

    The total points of a game run from about 125 to 229 and a bookmaker sets a different line for every one of them. A
    market whose line moves is not a column.
    """
    snapshots = source.to_snapshots(_payloads(source))
    assert not [col for col in snapshots.columns if col.startswith(('over_', 'under_'))]


def test_the_form_of_a_team_never_sees_its_own_game(source):
    """Test the form is what a team had done before a game, shifted so it can never contain the game's own result."""
    snapshots = source.to_snapshots(_payloads(source))
    preplay = snapshots[snapshots['event_status'] == 'preplay'].sort_values('date')
    first = preplay.iloc[0]
    assert pd.isna(first['home_points_for_avg'])

    later = preplay[preplay['home_team'] == 'Panathinaikos Athens'].iloc[0]
    assert later['home_points_for_avg'] == FORM_POINTS
    assert later['home_wins_avg'] == 1
