"""The reference venue, Betfair, over its official API.

It is the only surveyed exchange that carries a caller reference on a bet, and it takes two: `customerRef` de-duplicates
a resubmission for sixty seconds, and `customerOrderRef` persists and is filterable. Both are sent, with the same
derived value, so placing is once-only. There is no placement sandbox: the delayed key places real bets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from ._base import (
    BaseVenue,
    BetIdentity,
    ExecutionError,
    PlacementIntent,
    PlacementReceipt,
    PlacementStatus,
    VenueBlockedError,
)
from ._credentials import CredentialRef, resolve

API_URL = 'https://api.betfair.com/exchange/betting/json-rpc/v1'
ACCOUNT_URL = 'https://api.betfair.com/exchange/account/json-rpc/v1'
LOGIN_URL = 'https://identitysso-cert.betfair.com/api/certlogin'
BLOCKED_STATUS = frozenset({403, 429})
MARKET_COLS = ['match', 'market', 'selection', 'market_id', 'selection_id', 'price']

_ORDER_STATUS = {
    'EXECUTION_COMPLETE': PlacementStatus.MATCHED_FULL,
    'EXECUTABLE': PlacementStatus.ACCEPTED,
    'EXPIRED': PlacementStatus.REJECTED,
}


def _best_price(runner: dict[str, Any]) -> float | None:
    """Return the best price on offer for a runner."""
    available = runner.get('ex', {}).get('availableToBack', [])
    return float(available[0]['price']) if available else None


def _receipt(intent: PlacementIntent, report: dict[str, Any]) -> PlacementReceipt:
    """Return what the venue said about a placement."""
    instructions = report.get('instructionReports') or [{}]
    instruction = instructions[0]
    if report.get('status') != 'SUCCESS' or instruction.get('status') != 'SUCCESS':
        reason = instruction.get('errorCode') or report.get('errorCode') or 'the venue rejected the bet'
        return PlacementReceipt(
            identity=intent.identity,
            status=PlacementStatus.REJECTED,
            value_bet=intent.value_bet,
            detail=str(reason),
        )
    matched = float(instruction.get('sizeMatched', 0.0))
    status = _ORDER_STATUS.get(str(instruction.get('orderStatus')), PlacementStatus.ACCEPTED)
    if status is PlacementStatus.MATCHED_FULL and matched < intent.stake:
        status = PlacementStatus.MATCHED_PARTIAL
    return PlacementReceipt(
        identity=intent.identity,
        status=status,
        stake=matched or intent.stake,
        price=float(instruction.get('averagePriceMatched') or intent.min_price),
        venue_bet_id=instruction.get('betId'),
        value_bet=intent.value_bet,
        placed_at=pd.Timestamp.now(tz='UTC').to_pydatetime(),
    )


class BetfairVenue(BaseVenue):
    """The Betfair exchange, reached through its official betting API.

    Every credential is named rather than carried, so the defaults below are variable names and nothing sensitive has a
    default.

    Args:
        app_key_env:
            The name of the variable holding the application key.
        username_env:
            The name of the variable holding the username.
        password_env:
            The name of the variable holding the password.
        cert_path:
            The certificate the non-interactive login needs.
        cert_key_path:
            The key of that certificate.
        api_url:
            Where the betting calls go.
        account_url:
            Where the account calls go.
        login_url:
            Where the certificate login goes.
        min_interval:
            The seconds to leave between calls.

    Examples:
        >>> from sportsbet.execution import BetfairVenue
        >>> venue = BetfairVenue()
        >>> venue.key
        'betfair'
        >>> venue.can_cancel
        True
        >>> venue.app_key_env
        'BETFAIR_APP_KEY'
    """

    key = 'betfair'
    can_cancel = True

    def __init__(
        self: BetfairVenue,
        *,
        app_key_env: str = 'BETFAIR_APP_KEY',
        username_env: str = 'BETFAIR_USERNAME',
        password_env: str = 'BETFAIR_PASSWORD',  # noqa: S107
        cert_path: str | Path | None = None,
        cert_key_path: str | Path | None = None,
        api_url: str = API_URL,
        account_url: str = ACCOUNT_URL,
        login_url: str = LOGIN_URL,
        min_interval: float = 0.2,
    ) -> None:
        """Keep what the venue was configured with."""
        self.app_key_env = app_key_env
        self.username_env = username_env
        self.password_env = password_env
        self.cert_path = cert_path
        self.cert_key_path = cert_key_path
        self.api_url = api_url
        self.account_url = account_url
        self.login_url = login_url
        self.min_interval = min_interval
        self.token_: str | None = None

    def _ssl_context(self: BetfairVenue) -> ssl.SSLContext:
        """Return the context the certificate login needs."""
        if self.cert_path is None or self.cert_key_path is None:
            msg = 'The certificate login needs `cert_path` and `cert_key_path`.'
            raise ExecutionError(msg)
        context = ssl.create_default_context()
        context.load_cert_chain(certfile=str(self.cert_path), keyfile=str(self.cert_key_path))
        return context

    async def call(self: BetfairVenue, url: str, method: str, params: dict[str, Any]) -> Any:  # noqa: ANN401
        """Call the API and return what it answered.

        Args:
            url:
                Where the call goes.
            method:
                What to call.
            params:
                What to call it with.

        Returns:
            result:
                What the venue answered.
        """
        if self.token_ is None:
            msg = 'The venue is not authenticated. Call `authenticate` first.'
            raise ExecutionError(msg)
        headers = {
            'X-Application': resolve(CredentialRef(self.app_key_env)),
            'X-Authentication': self.token_,
            'Content-Type': 'application/json',
        }
        payload = {'jsonrpc': '2.0', 'method': method, 'params': params, 'id': 1}
        await asyncio.sleep(self.min_interval)
        async with aiohttp.ClientSession() as session, session.post(url, json=payload, headers=headers) as response:
            if response.status in BLOCKED_STATUS:
                msg = f'`{self.key}` refused the request with status {response.status}.'
                raise VenueBlockedError(msg)
            body = await response.json(content_type=None)
        if body.get('error'):
            msg = f'`{self.key}` answered with an error: {body["error"]}.'
            raise ExecutionError(msg)
        return body['result']

    async def authenticate(self: BetfairVenue) -> None:
        """Log in with the certificate and keep the session token."""
        data = {
            'username': resolve(CredentialRef(self.username_env)),
            'password': resolve(CredentialRef(self.password_env)),
        }
        headers = {'X-Application': resolve(CredentialRef(self.app_key_env))}
        async with (
            aiohttp.ClientSession() as session,
            session.post(self.login_url, data=data, headers=headers, ssl=self._ssl_context()) as response,
        ):
            if response.status in BLOCKED_STATUS:
                msg = f'`{self.key}` refused the login with status {response.status}.'
                raise VenueBlockedError(msg)
            body = await response.json(content_type=None)
        if body.get('loginStatus') != 'SUCCESS':
            msg = f'`{self.key}` refused the login: {body.get("loginStatus")}.'
            raise ExecutionError(msg)
        self.token_ = body['sessionToken']

    async def list_markets(self: BetfairVenue, matches: list[str]) -> pd.DataFrame:
        """Return the markets on offer for the given matches, with their current prices."""
        catalogue = await self.call(
            self.api_url,
            'SportsAPING/v1.0/listMarketCatalogue',
            {
                'filter': {'textQuery': ' '.join(matches)} if matches else {},
                'marketProjection': ['RUNNER_DESCRIPTION', 'EVENT'],
                'maxResults': 100,
            },
        )
        ids = [market['marketId'] for market in catalogue]
        if not ids:
            return pd.DataFrame(columns=MARKET_COLS)
        books = await self.call(
            self.api_url,
            'SportsAPING/v1.0/listMarketBook',
            {'marketIds': ids, 'priceProjection': {'priceData': ['EX_BEST_OFFERS']}},
        )
        prices = {
            (book['marketId'], runner['selectionId']): _best_price(runner)
            for book in books
            for runner in book.get('runners', [])
        }
        records = [
            {
                'match': market.get('event', {}).get('name', ''),
                'market': market.get('marketName', ''),
                'selection': runner.get('runnerName', ''),
                'market_id': market['marketId'],
                'selection_id': runner['selectionId'],
                'price': prices.get((market['marketId'], runner['selectionId'])),
            }
            for market in catalogue
            for runner in market.get('runners', [])
        ]
        return pd.DataFrame.from_records(records, columns=MARKET_COLS)

    async def read_balance(self: BetfairVenue) -> tuple[float, float]:
        """Return the balance and the exposure currently open."""
        funds = await self.call(self.account_url, 'AccountAPING/v1.0/getAccountFunds', {})
        return float(funds['availableToBetBalance']), abs(float(funds.get('exposure', 0.0)))

    async def _resolve(self: BetfairVenue, identity: BetIdentity) -> tuple[str, int] | None:
        """Return what the venue calls a market and a selection, which is not what a model calls them."""
        markets = await self.list_markets([identity.match])
        if markets.empty:
            return None
        wanted = markets[(markets['market'] == identity.market) & (markets['selection'] == identity.selection)]
        if wanted.empty:
            return None
        row = wanted.iloc[0]
        return str(row['market_id']), int(row['selection_id'])

    async def _orders(self: BetfairVenue, refs: list[str]) -> list[dict[str, Any]]:
        """Return the orders the venue holds for these references."""
        current = await self.call(self.api_url, 'SportsAPING/v1.0/listCurrentOrders', {'customerOrderRefs': refs})
        orders: list[dict[str, Any]] = current.get('currentOrders', [])
        return orders

    async def place(self: BetfairVenue, intent: PlacementIntent) -> PlacementReceipt:
        """Place one bet, once and only once for its identity.

        The reference goes on as both `customerRef` and `customerOrderRef`, and the request carries one instruction, so
        a resubmission inside sixty seconds is de-duplicated by the venue, and one after it is recognised by reading the
        order back.
        """
        ref = intent.identity.ref
        existing = await self._orders([ref])
        if existing:
            order = existing[0]
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.ALREADY_PLACED,
                stake=float(order.get('sizeMatched', 0.0)),
                price=float(order.get('averagePriceMatched') or 0.0) or None,
                venue_bet_id=order.get('betId'),
                value_bet=intent.value_bet,
                detail=f'`{self.key}` already holds a bet for this selection.',
            )
        resolved = await self._resolve(intent.identity)
        if resolved is None:
            return PlacementReceipt(
                identity=intent.identity,
                status=PlacementStatus.REJECTED,
                value_bet=intent.value_bet,
                detail=f'`{self.key}` offers no `{intent.identity.market}` on `{intent.identity.match}`.',
            )
        market_id, selection_id = resolved
        report = await self.call(
            self.api_url,
            'SportsAPING/v1.0/placeOrders',
            {
                'marketId': market_id,
                'customerRef': ref,
                'instructions': [
                    {
                        'selectionId': selection_id,
                        'side': 'BACK',
                        'orderType': 'LIMIT',
                        'customerOrderRef': ref,
                        'limitOrder': {'size': intent.stake, 'price': intent.min_price, 'persistenceType': 'LAPSE'},
                    },
                ],
            },
        )
        return _receipt(intent, report)

    async def read_status(self: BetfairVenue, identities: list[BetIdentity]) -> pd.DataFrame:
        """Return what the venue holds for these identities."""
        orders = await self._orders([identity.ref for identity in identities])
        return pd.DataFrame.from_records(orders)

    async def cancel(self: BetfairVenue, identity: BetIdentity) -> PlacementReceipt:
        """Cancel a bet."""
        orders = await self._orders([identity.ref])
        if not orders:
            return PlacementReceipt(
                identity=identity,
                status=PlacementStatus.REJECTED,
                detail=f'`{self.key}` holds no bet for this selection.',
            )
        order = orders[0]
        await self.call(
            self.api_url,
            'SportsAPING/v1.0/cancelOrders',
            {'marketId': order['marketId'], 'instructions': [{'betId': order['betId']}]},
        )
        return PlacementReceipt(
            identity=identity,
            status=PlacementStatus.REJECTED,
            venue_bet_id=order.get('betId'),
            detail=f'`{self.key}` cancelled the bet.',
        )
