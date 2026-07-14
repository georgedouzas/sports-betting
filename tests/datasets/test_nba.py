"""Tests for the statistics source of the NBA.

The feed is fetched on the user's machine and nothing is redistributed, so the transform is exercised against payloads
shaped like the real response.
"""

import json

import pandas as pd
import pytest

from sportsbet.sources import NBAStats, RawPayload

SNAPSHOTS = 9
MONTHS = 11
HOME_POINTS = 120
AWAY_POINTS = 96
FORM_POINTS = 120


def _event(date, home, away, home_points, away_points, *, played, season_type=2, competition='STD'):
    """Build an event the way the feed returns one."""
    return {
        'date': date,
        'name': f'{away} at {home}',
        'season': {'year': 2026, 'type': season_type},
        'competitions': [
            {
                'type': {'abbreviation': competition},
                'status': {'type': {'completed': played}},
                'competitors': [
                    {'homeAway': 'home', 'team': {'displayName': home}, 'score': str(home_points)},
                    {'homeAway': 'away', 'team': {'displayName': away}, 'score': str(away_points)},
                ],
            },
        ],
    }


SEASONS = json.dumps(
    {
        'count': 3,
        'items': [
            {'$ref': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2026?lang=en'},
            {'$ref': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2025?lang=en'},
            {'$ref': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024?lang=en'},
        ],
    },
).encode()

GAMES = json.dumps(
    {
        'events': [
            _event('2025-10-21T23:30Z', 'Boston Celtics', 'New York Knicks', 120, 96, played=True),
            _event('2025-10-22T23:00Z', 'Denver Nuggets', 'Los Angeles Lakers', 100, 110, played=True),
            _event('2025-10-25T23:00Z', 'Boston Celtics', 'Denver Nuggets', 105, 99, played=True),
            _event('2026-04-02T23:00Z', 'New York Knicks', 'Boston Celtics', 0, 0, played=False),
            _event(
                '2025-10-05T23:00Z',
                'Philadelphia 76ers',
                'Brooklyn Nets',
                88,
                84,
                played=True,
                season_type=1,
            ),
            _event(
                '2026-02-15T23:00Z',
                'Team Stripes',
                'Team Stars',
                40,
                35,
                played=True,
                season_type=2,
                competition='ALLSTAR',
            ),
            _event(
                '2026-02-15T20:00Z',
                'Team Stars',
                'World',
                31,
                29,
                played=True,
                season_type=2,
                competition='ALLSTAR',
            ),
            _event(
                '2026-04-20T23:00Z',
                'Los Angeles Lakers',
                'New York Knicks',
                115,
                108,
                played=True,
                season_type=3,
                competition='RD16',
            ),
        ],
    },
).encode()


@pytest.fixture
def source():
    """The statistics of the NBA."""
    return NBAStats()


def _payloads(source, content=GAMES):
    """Build the payload of a month the source declared."""
    item = source.required_items([{'league': 'NBA', 'division': 1, 'year': 2026}])[0]
    return [RawPayload(item=item, content=content)]


def test_the_catalogue_comes_from_the_feed(source):
    """Test the seasons are read from the feed, never fabricated from a range of years."""
    payloads = [RawPayload(item=source.index_items()[0], content=SEASONS)]
    assert source.catalogue(payloads) == [
        {'league': 'NBA', 'division': 1, 'year': 2024},
        {'league': 'NBA', 'division': 1, 'year': 2025},
        {'league': 'NBA', 'division': 1, 'year': 2026},
    ]


def test_a_season_is_named_by_the_year_it_ends_in_with_no_conversion(source):
    """Test the feed's season year is taken as it is.

    It already names a season by the year it ends in, so unlike the EuroLeague, whose `E2024` is the 2024-25 season,
    there is nothing to convert. Adding a year here would shift every season and empty the catalogue the odds share.
    """
    payloads = [RawPayload(item=source.index_items()[0], content=SEASONS)]
    assert {params['year'] for params in source.catalogue(payloads)} == {2024, 2025, 2026}


def test_the_catalogue_is_free(source):
    """Test the source charges nothing, so a preparation can be priced without spending anything."""
    assert source.estimate(source.index_items() + source.required_items([])) == 0


def test_a_season_is_asked_for_one_month_at_a_time(source):
    """Test the request window stays a month, which is the only reason a season comes back whole.

    The feed returns at most a thousand games and says nothing when it has more, while a season is about fourteen
    hundred. A month is two hundred and forty at its busiest. Widening this window would silently lose games, so this
    test exists to make widening it fail here rather than in a backtest.
    """
    items = source.required_items([{'league': 'NBA', 'division': 1, 'year': 2026}])
    assert len(items) == MONTHS
    windows = [item.url.split('dates=')[1].split('&')[0] for item in items]
    for window in windows:
        start, end = window.split('-')
        assert start[:6] == end[:6]
        assert start[6:] == '01'
    assert windows[0].startswith('202509')
    assert windows[-1].startswith('202607')


def test_the_tipoff_is_the_instant_the_feed_gives(source):
    """Test the tip-off is read as the UTC instant the feed publishes, never inferred from a local time."""
    snapshots = source.to_snapshots(_payloads(source))
    tipoffs = snapshots.set_index('home_team')['date']
    assert tipoffs['Boston Celtics'].iloc[0] == pd.Timestamp('2025-10-21 23:30')


def test_an_exhibition_is_excluded_even_though_it_is_filed_as_the_regular_season(source):
    """Test the all-star games never reach the dataset.

    The feed files them under the regular season, so keeping the regular season keeps them, and their teams are
    inventions. One in the roster breaks the pairing with the odds for the whole competition.
    """
    snapshots = source.to_snapshots(_payloads(source))
    teams = set(snapshots['home_team']) | set(snapshots['away_team'])
    assert not [team for team in teams if 'Team ' in team or team == 'World']


def test_a_playoff_game_is_included_even_though_it_is_not_a_standard_game(source):
    """Test the play-offs are kept.

    They are not filed as standard games, so keeping only the standard ones would drop every play-off game of the season
    and look entirely plausible.
    """
    snapshots = source.to_snapshots(_payloads(source))
    playoff = snapshots[
        (snapshots['home_team'] == 'Los Angeles Lakers') & (snapshots['away_team'] == 'New York Knicks')
    ]
    assert set(playoff['event_status']) == {'preplay', 'postplay'}


def test_the_preseason_is_excluded(source):
    """Test the pre-season never reaches the dataset."""
    snapshots = source.to_snapshots(_payloads(source))
    teams = set(snapshots['home_team']) | set(snapshots['away_team'])
    assert 'Philadelphia 76ers' not in teams
    assert 'Brooklyn Nets' not in teams


def test_an_unfamiliar_label_is_excluded_rather_than_admitted(source):
    """Test a competition the feed labels in a way we do not know is dropped, not kept.

    A missing game is a visible gap. An invented team in the roster is a silent corruption, so the rule is written to
    exclude rather than to admit.
    """
    content = json.dumps(
        {
            'events': [
                _event('2026-02-15T23:00Z', 'Team Moon', 'Team Sun', 50, 44, played=True, competition='ALLSTAR'),
                _event('2025-10-21T23:30Z', 'Boston Celtics', 'New York Knicks', 120, 96, played=True),
            ],
        },
    ).encode()
    snapshots = source.to_snapshots(_payloads(source, content))
    teams = set(snapshots['home_team']) | set(snapshots['away_team'])
    assert teams == {'Boston Celtics', 'New York Knicks'}


def test_a_game_that_was_played_carries_its_result(source):
    """Test a played game becomes a post-play snapshot with its final score."""
    snapshots = source.to_snapshots(_payloads(source))
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert postplay.loc['Boston Celtics', 'home_points'].iloc[0] == HOME_POINTS
    assert postplay.loc['Boston Celtics', 'away_points'].iloc[0] == AWAY_POINTS


def test_a_game_that_was_not_played_is_only_a_fixture(source):
    """Test an unplayed game gets a pre-play snapshot and nothing else, so it never becomes a training row.

    This is a game in a season whose other games are all finished, which is exactly the shape of a postponed game that
    was never made up. A finished season is not a season in which everything was played.
    """
    snapshots = source.to_snapshots(_payloads(source))
    assert len(snapshots) == SNAPSHOTS
    postponed = snapshots[(snapshots['home_team'] == 'New York Knicks') & (snapshots['away_team'] == 'Boston Celtics')]
    assert set(postponed['event_status']) == {'preplay'}


def test_the_outcome_of_a_game_that_cannot_be_drawn(source):
    """Test basketball has no draw, since a tie goes to overtime, so the outcome is two-way."""
    snapshots = source.to_snapshots(_payloads(source))
    assert 'draw' not in snapshots.columns
    postplay = snapshots[snapshots['event_status'] == 'postplay'].set_index('home_team')
    assert postplay.loc['Boston Celtics', 'home_win'].iloc[0] == 1
    assert postplay.loc['Boston Celtics', 'away_win'].iloc[0] == 0
    assert postplay.loc['Denver Nuggets', 'home_win'] == 0


def test_there_is_no_totals_market(source):
    """Test basketball has no totals line, so it has no totals market.

    A bookmaker sets a different line for every game, and a market whose line moves is not a column.
    """
    snapshots = source.to_snapshots(_payloads(source))
    assert not [col for col in snapshots.columns if col.startswith(('over_', 'under_'))]


def test_the_form_of_a_team_never_sees_its_own_game(source):
    """Test the form is what a team had done before a game, shifted so it can never contain the game's own result."""
    snapshots = source.to_snapshots(_payloads(source))
    preplay = snapshots[snapshots['event_status'] == 'preplay'].sort_values('date')
    assert pd.isna(preplay.iloc[0]['home_points_for_avg'])

    later = preplay[preplay['home_team'] == 'Boston Celtics'].iloc[1]
    assert later['home_points_for_avg'] == FORM_POINTS
    assert later['home_wins_avg'] == 1
