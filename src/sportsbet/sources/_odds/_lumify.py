"""Read live and upcoming odds from Lumify."""

# License: MIT


from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp
import pandas as pd

from .._base import CONNECTIONS_LIMIT, BaseOddsSource, RawItem, RawPayload

URL = 'https://lumify.ai'
EVENTS_URL = f'{URL}/v1/events'
# Lumify sport/league slug → library (league, division). sport is None carries every mapped competition.
LEAGUES_MAPPING: dict[tuple[str, str | None], tuple[str, int]] = {
    ('nba', None): ('NBA', 1),
    ('soccer', 'epl'): ('England', 1),
    ('soccer', 'la_liga'): ('Spain', 1),
    ('soccer', 'serie_a'): ('Italy', 1),
    ('soccer', 'bundesliga'): ('Germany', 1),
    ('soccer', 'ligue_1'): ('France', 1),
    ('soccer', 'mls'): ('USA', 1),
}
MARKETS = ['h2h', 'totals']
BOOKMAKERS = ['pinnacle']
TOTALS_POINT = 2.5
DELIMITER = '__'
LIVE_KEY = 'live'
USER_AGENT = 'sportsbet-lumify-odds/0.1'


def _now() -> pd.Timestamp:
    """Return the current instant."""
    return pd.Timestamp(datetime.now(tz=UTC))


def _american_to_decimal(price: float | int) -> float:
    """Convert American odds to European decimal odds."""
    american = float(price)
    if american >= 100:
        return round(1.0 + american / 100.0, 4)
    return round(1.0 + 100.0 / abs(american), 4)


def _normalize(name: str) -> str:
    """Return a comparison key for a team or outcome label."""
    return ''.join(character for character in (name or '').lower() if character.isalnum())


def _participants(event: dict) -> tuple[str, str]:
    """Return the home and away names from a Lumify event."""
    home = away = ''
    for participant in event.get('participants') or []:
        role = (participant.get('role') or '').lower()
        name = (
            (participant.get('team') or {}).get('name')
            or (participant.get('player') or {}).get('name')
            or participant.get('name')
            or ''
        )
        if role == 'home':
            home = name
        elif role == 'away':
            away = name
    return home, away


def _parse_key(key: str) -> tuple[str, str | None, int]:
    """Return the sport, optional league slug, and year an item key encodes."""
    sport, league, year, _live = key.split(DELIMITER)
    return sport, None if league == '-' else league, int(year)


def _outcomes(bookmaker: dict, home_team: str, away_team: str, markets: list[str]) -> dict[str, float]:
    """Return the markets of a bookmaker, named the way the library names them."""
    home_n, away_n = _normalize(home_team), _normalize(away_team)
    outcomes: dict[str, float] = {}
    for market in bookmaker.get('markets') or []:
        market_key = (market.get('key') or market.get('label') or '').lower()
        for outcome in market.get('outcomes') or []:
            label = outcome.get('outcome') or outcome.get('name') or ''
            price = outcome.get('price')
            if price is None:
                continue
            decimal = _american_to_decimal(price)
            label_n = _normalize(label)
            if market_key in {'h2h', 'moneyline'} and 'h2h' in markets:
                if label_n in {'home', '1'} or (home_n and (label_n == home_n or home_n in label_n or label_n in home_n)):
                    outcomes['home_win'] = decimal
                elif label_n in {'away', '2'} or (
                    away_n and (label_n == away_n or away_n in label_n or label_n in away_n)
                ):
                    outcomes['away_win'] = decimal
                elif label_n in {'draw', 'x', 'tie'}:
                    outcomes['draw'] = decimal
            elif market_key in {'totals', 'total'} and 'totals' in markets:
                point = outcome.get('point')
                if point is None or float(point) != TOTALS_POINT:
                    continue
                if label_n.startswith('over') or label_n == 'over':
                    outcomes['over_2.5'] = decimal
                elif label_n.startswith('under') or label_n == 'under':
                    outcomes['under_2.5'] = decimal
    return outcomes


class LumifyOdds(BaseOddsSource):
    """The live and upcoming odds of [Lumify](https://lumify.ai), an agent-ready sports intelligence API.

    Lumify prices the current slate (scheduled and in-progress events). It does not sell historical closing lines the
    way The Odds API does, so it pairs best with fixture extraction and the season in progress.

    The key is read from the environment variable named by `key_env` and sent as an `Authorization: Bearer` header. It
    never enters a `RawItem`.

    Free instant trial keys (no signup): https://lumify.ai/docs/ai

    Args:
        key_env:
            The name of the environment variable holding your API key.

        markets:
            The markets to price, e.g. `['h2h', 'totals']`. The default `None` uses both.

        bookmakers:
            The bookmakers to request, e.g. `['pinnacle', 'draftkings']`. The default `None` uses `['pinnacle']`.
            Pass `['all']` for every book Lumify carries (costs more credits per event).

    Examples:
        >>> import os
        >>> from sportsbet.sources import LumifyOdds, RawItem
        >>> os.environ['LUMIFY_API_KEY'] = 'secret'
        >>> source = LumifyOdds(key_env='LUMIFY_API_KEY', markets=['h2h'])
        >>> source.name, source.kind
        ('lumify', 'odds')
        >>> source.sport is None
        True
        >>> item = RawItem(source='lumify', key='nba__-__2026__live', url='https://lumify.ai/v1/events?sport=nba')
        >>> 'secret' in item.url
        False
        >>> source.request_headers(item)['Authorization']
        'Bearer secret'
    """

    name: ClassVar[str] = 'lumify'

    def __init__(
        self: Self,
        key_env: str,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ) -> None:
        self.key_env = key_env
        self.markets = markets
        self.bookmakers = bookmakers

    def _settings(self: Self) -> tuple[list[str], list[str]]:
        """Return the markets and bookmakers, defaulted."""
        markets = self.markets if self.markets is not None else MARKETS
        bookmakers = self.bookmakers if self.bookmakers is not None else BOOKMAKERS
        return markets, bookmakers

    def request_headers(self: Self, item: RawItem) -> dict[str, str]:
        """Return the Bearer authorization header, reading the key from the environment."""
        return {
            'Authorization': f'Bearer {os.environ[self.key_env]}',
            'Accept': 'application/json',
            'User-Agent': USER_AGENT,
        }

    def list_index_items(self: Self, selection: Any = None) -> list[RawItem]:
        """Return no index items; the catalogue is the fixed league map."""
        return []

    def read_catalogue(self: Self, payloads: list[RawPayload]) -> list[dict]:
        """Return the mapped leagues for the current and next calendar year."""
        years = range(_now().year, _now().year + 2)
        return [
            {'league': league, 'division': division, 'year': year}
            for (_sport, _league_slug), (league, division) in LEAGUES_MAPPING.items()
            for year in years
        ]

    def list_required_items(self: Self, params: list[dict], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """Return one live-slate item per selected Lumify sport and league."""
        reverse = {
            (league, division): (sport, league_slug) for (sport, league_slug), (league, division) in LEAGUES_MAPPING.items()
        }
        items: list[RawItem] = []
        seen: set[str] = set()
        for param in params:
            mapped = reverse.get((param['league'], param['division']))
            if mapped is None:
                continue
            sport, league_slug = mapped
            key = DELIMITER.join([sport, league_slug or '-', str(param['year']), LIVE_KEY])
            if key in seen:
                continue
            seen.add(key)
            query: dict[str, str] = {'sport': sport, 'limit': '50'}
            if league_slug:
                query['league'] = league_slug
            items.append(RawItem(source=self.name, key=key, url=f'{EVENTS_URL}?{urlencode(query)}'))
        return items

    def fetch_items(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """List the slate for each item, then fetch each event with odds inlined."""
        if not items:
            return []
        return asyncio.run(self._fetch_slates(items))

    async def _fetch_slates(self: Self, items: list[RawItem]) -> list[RawPayload]:
        """Fetch every slate."""
        headers = self.request_headers(items[0])
        _, bookmakers = self._settings()
        bookmaker = ','.join(bookmakers)
        async with aiohttp.ClientSession(
            raise_for_status=True,
            connector=aiohttp.TCPConnector(limit=CONNECTIONS_LIMIT),
            headers=headers,
        ) as client:
            payloads = []
            for item in items:
                events = await self._fetch_slate(client, item, bookmaker)
                payloads.append(RawPayload(item=item, content=json.dumps(events).encode('utf-8')))
            return payloads

    async def _fetch_slate(
        self: Self,
        client: aiohttp.ClientSession,
        item: RawItem,
        bookmaker: str,
    ) -> list[dict]:
        """Return events with odds for one sport/league slate."""
        base_query = {key: values[0] for key, values in parse_qs(urlparse(item.url).query).items()}
        event_ids: list[int] = []
        for status in ('scheduled', 'inprogress'):
            after_id: int | None = None
            while True:
                query = {**base_query, 'status': status, 'limit': base_query.get('limit', '50')}
                if after_id is not None:
                    query['after_id'] = str(after_id)
                async with client.get(EVENTS_URL, params=query) as response:
                    payload = await response.json(content_type=None)
                batch = payload.get('events') or []
                for event in batch:
                    if event.get('id') is not None:
                        event_ids.append(int(event['id']))
                after_id = payload.get('next_after_id')
                if not batch or after_id is None:
                    break

        events: list[dict] = []
        for event_id in event_ids:
            params = {'include_odds': 'true', 'bookmaker': bookmaker}
            async with client.get(f'{EVENTS_URL}/{event_id}', params=params) as response:
                events.append(await response.json(content_type=None))
        return events

    def to_snapshots(self: Self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform Lumify event+odds payloads into long odds snapshots."""
        markets, _ = self._settings()
        records: list[dict] = []
        for payload in payloads:
            sport, _league_slug, year = _parse_key(payload.item.key)
            league, division = LEAGUES_MAPPING[(sport, _league_slug)]
            events = json.loads(payload.content.decode('utf-8'))
            for event in events:
                home_team, away_team = _participants(event)
                if not home_team or not away_team:
                    continue
                kickoff = event.get('starts_at') or event.get('scheduled_start_at')
                if not kickoff:
                    continue
                date = pd.Timestamp(kickoff)
                if date.tzinfo is None:
                    date = date.tz_localize('UTC')
                else:
                    date = date.tz_convert('UTC')
                status = (event.get('status') or 'scheduled').lower()
                event_status = 'inplay' if status == 'inprogress' else 'preplay'
                odds_payload = event.get('odds') or {}
                for bookmaker in odds_payload.get('bookmakers') or []:
                    priced = _outcomes(bookmaker, home_team, away_team, markets)
                    if not priced:
                        continue
                    records.append(
                        {
                            'event_status': event_status,
                            'event_time': 0,
                            'date': date,
                            'league': league,
                            'division': division,
                            'year': year,
                            'home_team': home_team,
                            'away_team': away_team,
                            'provider': bookmaker.get('bookmaker') or 'unknown',
                            **priced,
                        },
                    )
        return pd.DataFrame(records)
