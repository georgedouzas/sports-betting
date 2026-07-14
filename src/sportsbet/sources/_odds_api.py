"""Implements the odds source backed by The Odds API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from urllib.parse import urlencode

import pandas as pd

from .. import ParamGrid
from ._base import BaseOddsSource, RawItem, RawPayload
from ._schema import EVENT_COLS

URL = 'https://api.the-odds-api.com/v4'
SPORTS_URL = f'{URL}/sports'
HISTORICAL_URL = f'{URL}/historical/sports/{{sport}}/odds'
LIVE_URL = f'{URL}/sports/{{sport}}/odds'

LEAGUES_MAPPING = {
    'soccer_epl': ('England', 1),
    'soccer_efl_champ': ('England', 2),
    'soccer_england_league1': ('England', 3),
    'soccer_england_league2': ('England', 4),
    'soccer_spl': ('Scotland', 1),
    'soccer_germany_bundesliga': ('Germany', 1),
    'soccer_germany_bundesliga2': ('Germany', 2),
    'soccer_italy_serie_a': ('Italy', 1),
    'soccer_italy_serie_b': ('Italy', 2),
    'soccer_spain_la_liga': ('Spain', 1),
    'soccer_spain_segunda_division': ('Spain', 2),
    'soccer_france_ligue_one': ('France', 1),
    'soccer_france_ligue_two': ('France', 2),
    'soccer_netherlands_eredivisie': ('Netherlands', 1),
    'soccer_belgium_first_div': ('Belgium', 1),
    'soccer_portugal_primeira_liga': ('Portugal', 1),
    'soccer_turkey_super_league': ('Turkey', 1),
    'soccer_greece_super_league': ('Greece', 1),
    'soccer_argentina_primera_division': ('Argentina', 1),
    'soccer_austria_bundesliga': ('Austria', 1),
    'soccer_brazil_campeonato': ('Brazil', 1),
    'soccer_china_superleague': ('China', 1),
    'soccer_denmark_superliga': ('Denmark', 1),
    'soccer_finland_veikkausliiga': ('Finland', 1),
    'soccer_league_of_ireland': ('Ireland', 1),
    'soccer_japan_j_league': ('Japan', 1),
    'soccer_mexico_ligamx': ('Mexico', 1),
    'soccer_norway_eliteserien': ('Norway', 1),
    'soccer_poland_ekstraklasa': ('Poland', 1),
    'soccer_russia_premier_league': ('Russia', 1),
    'soccer_sweden_allsvenskan': ('Sweden', 1),
    'soccer_switzerland_superleague': ('Switzerland', 1),
    'soccer_usa_mls': ('USA', 1),
    'basketball_euroleague': ('Euroleague', 1),
    'basketball_nba': ('NBA', 1),
    'basketball_wnba': ('WNBA', 1),
    'basketball_ncaab': ('NCAAB', 1),
    'basketball_nbl': ('NBL', 1),
}
MARKETS = ['h2h', 'totals']
REGIONS = ['eu']
MOMENTS = [('preplay', 0), ('inplay', 45)]
CLOSING_OFFSET = pd.Timedelta(minutes=-1)
SNAPSHOT_TOLERANCE = pd.Timedelta(minutes=5)
TOTALS_POINT = 2.5
HISTORICAL_MULTIPLIER = 10
HISTORICAL_START = pd.Timestamp('2020-06-06', tz='UTC')
FIRST_YEAR = 2021
SPORTS_KEY = 'sports'
LIVE_KEY = 'live'
DELIMITER = '__'
SNAPSHOT_MINUTES = 5


def _now() -> pd.Timestamp:
    """Return the current instant."""
    return pd.Timestamp(datetime.now(tz=UTC))


def _timestamp(moment: pd.Timestamp) -> str:
    """Render an instant the way the vendor addresses its snapshots."""
    return moment.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def _key_timestamp(moment: pd.Timestamp) -> str:
    """Render an instant so it can be part of an item key, which becomes a file name."""
    return moment.tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')


def _events(payload: RawPayload) -> tuple[list[dict], str]:
    """Return the events of a payload and the endpoint they came from."""
    content: Any = json.loads(payload.content)
    if isinstance(content, dict):
        return content.get('data', []), 'historical'
    return content, 'live'


class OddsApi(BaseOddsSource):
    """The odds of The Odds API.

    It carries time-stamped prices, so an in-play bet can be backtested against the odds that were actually available at
    the minute it would have been placed. The free feed cannot do that: it only publishes the closing price.

    It needs your own key, and the data it buys never leaves your machine. The key is added to a request when the
    request is made, so it is never written to the store.

    Historical prices are a paid tier and begin on 6 June 2020. A historical snapshot costs ten times a live one, so the
    preparation prices the work before it is done: run it with `dry_run=True` first.

    Read more in the [user guide][user-guide].

    Args:
        key:
            Your API key.

        markets:
            The markets to price, e.g. `['h2h', 'totals']`. The default `None` uses both.

        regions:
            The bookmaker regions, e.g. `['eu', 'uk']`. The default `None` uses `['eu']`. Every region multiplies the
            cost.

        moments:
            The moments of a match to price, as `(event_status, minutes)` pairs.
            The default `None` prices the moments the statistics carry, and no
            others: a price for a moment there is nothing to pair with can never
            be used, and it costs the same as one that can. Every moment is a
            separate snapshot, so it multiplies the cost.
    """

    name: ClassVar[str] = 'odds_api'

    def __init__(
        self: Self,
        key: str,
        markets: list[str] | None = None,
        regions: list[str] | None = None,
        moments: list[tuple[str, int]] | None = None,
    ) -> None:
        self.key = key
        self.markets = markets
        self.regions = regions
        self.moments = moments

    def _settings(self: Self) -> tuple[list[str], list[str], list[tuple[str, int]]]:
        """Return the markets, regions and moments, defaulted."""
        markets = self.markets if self.markets is not None else MARKETS
        regions = self.regions if self.regions is not None else REGIONS
        moments = self.moments if self.moments is not None else MOMENTS
        return markets, regions, moments

    @staticmethod
    def _moments(schedule: pd.DataFrame) -> list[tuple[str, int]]:
        """Return the moments the statistics carry, which are the only ones worth a price.

        A price is bought to be paired with what was known at that moment, so a moment the statistics do not have is a
        price that can never be used — and it costs the same as one that can. A sport whose statistics stop at the
        whistle has no use for the odds at half time.
        """
        moments = schedule[list(EVENT_COLS)].drop_duplicates()
        return sorted(
            {
                (event_status, int(pd.Timedelta(event_time).total_seconds() // 60))
                for event_status, event_time in moments.itertuples(index=False)
                if event_status != 'postplay'
            },
        )

    def _query(self: Self) -> dict[str, str]:
        """Return the query the vendor expects, without the credential."""
        markets, regions, _ = self._settings()
        return {'regions': ','.join(regions), 'markets': ','.join(markets), 'oddsFormat': 'decimal'}

    def request_url(self: Self, item: RawItem) -> str:
        """Return the URL to fetch an item from, with the key added.

        The key is added here and nowhere else, so it never reaches a `RawItem` and is never written to the store.

        Args:
            item:
                The item to fetch.

        Returns:
            url:
                Where to fetch it from.
        """
        separator = '&' if '?' in item.url else '?'
        return f'{item.url}{separator}apiKey={self.key}'

    def needs_schedule(self: Self) -> bool:
        """Return that the source has to be told when the matches are.

        Its prices are addressed by instant rather than by season, so `kick-off + 45min` is a timestamp it can only
        build once it knows the kick-off.

        Returns:
            needed:
                Always `True`.
        """
        return True

    def index_items(self: Self, selection: ParamGrid | None = None) -> list[RawItem]:
        """Return the catalogue of the vendor, which is free."""
        return [RawItem(source=self.name, key=SPORTS_KEY, url=f'{SPORTS_URL}?all=true', volatile=True)]

    def catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the combinations the vendor covers.

        The vendor has no notion of a season, so the years are its historical coverage. A competition it does not list,
        or one that is not mapped to a league, is left out rather than guessed at.
        """
        if not payloads:
            return []
        sports = json.loads(payloads[0].content)
        years = range(FIRST_YEAR, _now().year + 2)
        return [
            {'league': league, 'division': division, 'year': year}
            for sport in sports
            if (mapped := LEAGUES_MAPPING.get(sport['key'])) is not None
            for league, division in [mapped]
            for year in years
        ]

    def required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one item per snapshot the selected matches need.

        A snapshot holds every match priced at that instant, so matches that kick off together share an item and are
        paid for once.

        Args:
            params:
                The selected parameter combinations.

            schedule:
                The matches of the selected parameters, with their kick-off instants.

        Returns:
            items:
                The snapshots to fetch.
        """
        if schedule is None or schedule.empty:
            return []
        markets, regions, moments = self._settings()
        if self.moments is None:
            moments = self._moments(schedule)
        sports = {(league, division): sport for sport, (league, division) in LEAGUES_MAPPING.items()}
        cost = HISTORICAL_MULTIPLIER * len(markets) * len(regions)
        now = _now()

        items: list[RawItem] = []
        for (league, division, year), matches in schedule.groupby(['league', 'division', 'year']):
            sport = sports.get((league, division))
            if sport is None:
                continue
            for event_status, minutes in moments:
                offset = CLOSING_OFFSET if event_status == 'preplay' else pd.Timedelta(minutes=minutes)
                snapshots = (matches['date'] + offset).drop_duplicates()
                snapshots = snapshots[(snapshots >= HISTORICAL_START) & (snapshots < now)]
                for snapshot in sorted(snapshots):
                    query = urlencode({**self._query(), 'date': _timestamp(snapshot)})
                    url = f'{HISTORICAL_URL.format(sport=sport)}?{query}'
                    key = DELIMITER.join([sport, str(year), _key_timestamp(snapshot), event_status, str(minutes)])
                    items.append(RawItem(source=self.name, key=key, url=url, cost=cost))
            live = f'{LIVE_URL.format(sport=sport)}?{urlencode(self._query())}'
            live_key = DELIMITER.join([sport, str(year), LIVE_KEY])
            live_cost = len(markets) * len(regions)
            items.append(RawItem(source=self.name, key=live_key, url=live, volatile=True, cost=live_cost))
        return items

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the vendor's responses into the long odds snapshots.

        A snapshot holds every match the vendor priced at that instant, so only the matches the item was asked for are
        kept: the others are priced by their own item, at their own moment.

        Args:
            payloads:
                The payloads of the required items.

        Returns:
            snapshots:
                The long odds snapshots.
        """
        records: list[dict] = []
        for payload in payloads:
            events, endpoint = _events(payload)
            if endpoint == 'historical':
                records.extend(self._historical_records(payload, events))
            else:
                records.extend(self._live_records(payload, events))
        return pd.DataFrame(records)

    def _historical_records(self: Self, payload: RawPayload, events: list[dict]) -> list[dict]:
        """Return the odds of the matches an historical snapshot was asked for."""
        sport, year, snapshot, event_status, minutes = _parse_key(payload.item.key)
        offset = CLOSING_OFFSET if event_status == 'preplay' else pd.Timedelta(minutes=minutes)
        league, division = LEAGUES_MAPPING[sport]
        records = []
        for event in events:
            kickoff = pd.Timestamp(event['commence_time'])
            if snapshot is None or abs((kickoff + offset) - snapshot) > SNAPSHOT_TOLERANCE:
                continue
            records.extend(self._event_records(event, kickoff, league, division, year, event_status, minutes))
        return records

    def _live_records(self: Self, payload: RawPayload, events: list[dict]) -> list[dict]:
        """Return the odds of the matches the live endpoint priced.

        A match that has kicked off is priced at the minute it has reached, so an upcoming and a running match are both
        carried by the same request.
        """
        sport, year, *_ = _parse_key(payload.item.key)
        league, division = LEAGUES_MAPPING[sport]
        now = _last_update(events)
        records = []
        for event in events:
            kickoff = pd.Timestamp(event['commence_time'])
            elapsed = (now - kickoff).total_seconds() / 60 if now is not None else 0
            event_status, minutes = ('inplay', _round_minutes(elapsed)) if elapsed > 0 else ('preplay', 0)
            records.extend(self._event_records(event, kickoff, league, division, year, event_status, minutes))
        return records

    def _event_records(
        self: Self,
        event: dict,
        kickoff: pd.Timestamp,
        league: str,
        division: int,
        year: int,
        event_status: str,
        minutes: int,
    ) -> list[dict]:
        """Return one row per bookmaker of a match, with its markets as columns."""
        records = []
        for bookmaker in event.get('bookmakers', []):
            outcomes = _outcomes(bookmaker, event['home_team'], event['away_team'])
            if not outcomes:
                continue
            records.append(
                {
                    'event_status': event_status,
                    'event_time': minutes,
                    'date': kickoff,
                    'league': league,
                    'division': division,
                    'year': year,
                    'home_team': event['home_team'],
                    'away_team': event['away_team'],
                    'provider': bookmaker['key'],
                    **outcomes,
                },
            )
        return records


def _parse_key(key: str) -> tuple[str, int, pd.Timestamp | None, str, int]:
    """Return the sport, the season, the instant and the moment an item key encodes.

    The season comes from the statistics rather than from the kick-off, since a league played over a calendar year names
    its seasons differently from one played across two.
    """
    parts = key.split(DELIMITER)
    if parts[-1] == LIVE_KEY:
        sport, year, _ = parts
        return sport, int(year), None, 'preplay', 0
    sport, year, snapshot, event_status, minutes = parts
    return sport, int(year), pd.Timestamp(snapshot, tz='UTC'), event_status, int(minutes)


def _last_update(events: list[dict]) -> pd.Timestamp | None:
    """Return the latest instant the vendor priced anything at, which stands in for now."""
    updates = [
        pd.Timestamp(bookmaker['last_update'])
        for event in events
        for bookmaker in event.get('bookmakers', [])
        if bookmaker.get('last_update')
    ]
    return max(updates) if updates else None


def _round_minutes(elapsed: float) -> int:
    """Round an elapsed time to the granularity the vendor prices at."""
    return int(round(elapsed / SNAPSHOT_MINUTES) * SNAPSHOT_MINUTES)


def _outcomes(bookmaker: dict, home_team: str, away_team: str) -> dict:
    """Return the markets of a bookmaker, named the way the library names them."""
    outcomes: dict[str, float] = {}
    for market in bookmaker.get('markets', []):
        for outcome in market.get('outcomes', []):
            name, price, point = outcome.get('name'), outcome.get('price'), outcome.get('point')
            if market['key'] == 'h2h':
                if name == home_team:
                    outcomes['home_win'] = price
                elif name == away_team:
                    outcomes['away_win'] = price
                elif name == 'Draw':
                    outcomes['draw'] = price
            elif market['key'] == 'totals' and point == TOTALS_POINT:
                if name == 'Over':
                    outcomes['over_2.5'] = price
                elif name == 'Under':
                    outcomes['under_2.5'] = price
    return outcomes
