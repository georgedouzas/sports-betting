"""Implements the server that lets an agent drive the library.

The agent lives outside the library. The library stays a set of estimators that behave the same way every time they are
run; the agent calls them, holds the keys and makes the choices, and the server connects the two.

A tool is told what to do in its arguments, exactly as a command is, so everything it needs is in the call. The argument
names the environment variable holding a key, so the key itself stays out of a transcript.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
from mcp.server.fastmcp import FastMCP
from sklearn.model_selection import TimeSeriesSplit

from .._artifacts import load_dataloader, save_dataloader
from .._selection import DEFAULT_KEY_ENV, build_bettor, build_dataloader, build_venue
from ..evaluation import backtest as run_backtest
from ..evaluation import load_bettor, save_bettor
from ..execution import (
    BaseVenue,
    BetIdentity,
    BrowserSession,
    ExposureLimits,
    PlacementIntent,
    PlacementQuote,
    value_bet_intents,
)
from ..execution import execute as run_execute
from ..execution import place as run_place
from ..execution import quote as run_quote

server: FastMCP = FastMCP('sportsbet')

Answer = TypeVar('Answer')
Selection = dict[str, Any]
Strategy = dict[str, Any]


async def _offload(work: Callable[..., Answer], *args: object) -> Answer:
    """Run the library in a thread, since it fetches with an event loop of its own.

    A tool is answered inside an event loop, and the library opens one to fetch. A loop cannot be opened inside a loop,
    so anything that might fetch is handed to a thread that has none.
    """
    return await asyncio.to_thread(work, *args)


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Return a frame as records, which an agent can read."""
    if frame is None or frame.empty:
        return []
    return [
        {col: (None if pd.isna(value) else value) for col, value in row.items()}
        for row in frame.astype(object).to_dict(orient='records')
    ]


def _selection(
    stats: str,
    odds: str | None,
    leagues: list[str] | None,
    divisions: list[int] | None,
    years: list[int] | None,
    odds_key_env: str,
    odds_markets: list[str] | None,
    odds_regions: list[str] | None,
    odds_moments: list[str] | None,
    aliases: list[str] | None,
    max_unmatched_rate: float,
) -> Selection:
    """Return what a tool was told about the data to use."""
    return {
        'leagues': leagues,
        'divisions': divisions,
        'years': years,
        'stats': stats,
        'odds': odds,
        'odds_key_env': odds_key_env,
        'odds_markets': odds_markets,
        'odds_regions': odds_regions,
        'odds_moments': odds_moments,
        'aliases': aliases,
        'max_unmatched_rate': max_unmatched_rate,
    }


def _extraction(
    odds_type: str | None,
    drop_na_thres: float | None,
    target_event_status: str | None,
    target_event_time: str | None,
    input_event_status: str | None,
    input_event_time: str | None,
) -> dict[str, Any]:
    """Return how a tool was told to extract, which is what decides the moment a model bets at."""
    settings: dict[str, Any] = {
        'odds_type': odds_type,
        'drop_na_thres': drop_na_thres,
        'target_event_status': target_event_status,
        'input_event_status': input_event_status,
    }
    for name, value in (('target_event_time', target_event_time), ('input_event_time', input_event_time)):
        if value is not None:
            settings[name] = pd.Timedelta(value)
    return {name: value for name, value in settings.items() if value is not None}


def _strategy(
    model: str,
    alpha: float,
    betting_markets: list[str] | None,
    init_cash: float | None,
    stake: float | None,
    model_odds_types: list[str] | None,
) -> Strategy:
    """Return what a tool was told about the betting model to use."""
    return {
        'model': model,
        'alpha': alpha,
        'betting_markets': betting_markets,
        'init_cash': init_cash,
        'stake': stake,
        'model_odds_types': model_odds_types,
    }


def _available_params(selection: Selection) -> list[dict]:
    """Return what can be selected."""
    stats_source, *_ = build_dataloader(**selection).sources
    return stats_source.available_params()


def _odds_types(selection: Selection) -> list[str]:
    """Return the odds types a selection carries."""
    return list(build_dataloader(**selection).get_odds_types())


def _extract_train_data(selection: Selection, extraction: dict[str, Any], output: str | None) -> dict[str, Any]:
    """Download the training data, and write the dataloader where the other tools can read it.

    This is the only tool that downloads the seasons. Everything after it reads what this wrote, so a metered feed is
    bought once rather than once per call.
    """
    dataloader = build_dataloader(**selection)
    X, Y, O = dataloader.extract_train_data(**extraction)
    if output is not None:
        save_dataloader(output, dataloader, (X, Y, O))
    return {'X': _records(X), 'Y': _records(Y), 'O': _records(O), 'output': output}


def _extract_exploration_data(selection: Selection, extraction: dict[str, Any]) -> dict[str, Any]:
    """Return the features on their own, with no targets and no odds."""
    settings = {name: value for name, value in extraction.items() if name != 'odds_type'}
    X = build_dataloader(**selection).extract_exploration_data(**settings)
    return {'X': _records(X)}


def _extract_fixtures_data(dataloader: str) -> dict[str, Any]:
    """Return the games that have not been played yet, from a saved dataloader."""
    loader, _ = load_dataloader(dataloader)
    X, _, O = loader.extract_fixtures_data()
    return {'X': _records(X), 'O': _records(O)}


def _backtest(dataloader: str, strategy: Strategy, cv: int, n_jobs: int, verbose: int) -> list[dict[str, Any]]:
    """Return the backtesting results of a model on a saved dataloader."""
    _, (X, Y, O) = load_dataloader(dataloader)
    bettor = build_bettor(**strategy)
    return _records(run_backtest(bettor, X, Y, O, cv=TimeSeriesSplit(cv), n_jobs=n_jobs, verbose=verbose))


def _fit(dataloader: str, strategy: Strategy, output: str) -> dict[str, Any]:
    """Fit a model on a saved dataloader and save it, so it is fitted once and reused."""
    _, (X, Y, O) = load_dataloader(dataloader)
    bettor = build_bettor(**strategy)
    bettor.fit(X, Y, O)
    save_bettor(bettor, output)
    return {'output': output, 'betting_markets': list(bettor.betting_markets_)}


def _bet(dataloader: str, bettor: str) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
    loader, _ = load_dataloader(dataloader)
    fitted = load_bettor(bettor)
    X_fix, _, O_fix = loader.extract_fixtures_data()
    if X_fix.empty or O_fix is None or O_fix.empty:
        return []
    value_bets = pd.DataFrame(fitted.bet(X_fix, O_fix), columns=list(fitted.betting_markets_))
    games = X_fix[['home_team', 'away_team']].reset_index()
    return _records(pd.concat([games, value_bets], axis=1))


@server.tool()
async def available_params(
    stats: str,
    odds: str | None = None,
    leagues: list[str] | None = None,
    divisions: list[int] | None = None,
    years: list[int] | None = None,
    odds_key_env: str = DEFAULT_KEY_ENV,
    odds_markets: list[str] | None = None,
    odds_regions: list[str] | None = None,
    odds_moments: list[str] | None = None,
    aliases: list[str] | None = None,
    max_unmatched_rate: float = 0.0,
) -> list[dict]:
    """Return the leagues, divisions and seasons that can be selected."""
    selection = _selection(
        stats,
        odds,
        leagues,
        divisions,
        years,
        odds_key_env,
        odds_markets,
        odds_regions,
        odds_moments,
        aliases,
        max_unmatched_rate,
    )
    result: list[dict] = await _offload(_available_params, selection)
    return result


@server.tool()
async def odds_types(
    stats: str,
    odds: str | None = None,
    leagues: list[str] | None = None,
    divisions: list[int] | None = None,
    years: list[int] | None = None,
    odds_key_env: str = DEFAULT_KEY_ENV,
    odds_markets: list[str] | None = None,
    odds_regions: list[str] | None = None,
    odds_moments: list[str] | None = None,
    aliases: list[str] | None = None,
    max_unmatched_rate: float = 0.0,
) -> list[str]:
    """Return the odds types a selection carries."""
    selection = _selection(
        stats,
        odds,
        leagues,
        divisions,
        years,
        odds_key_env,
        odds_markets,
        odds_regions,
        odds_moments,
        aliases,
        max_unmatched_rate,
    )
    result: list[str] = await _offload(_odds_types, selection)
    return result


@server.tool()
async def extract_train_data(
    stats: str,
    odds: str | None = None,
    leagues: list[str] | None = None,
    divisions: list[int] | None = None,
    years: list[int] | None = None,
    odds_key_env: str = DEFAULT_KEY_ENV,
    odds_markets: list[str] | None = None,
    odds_regions: list[str] | None = None,
    odds_moments: list[str] | None = None,
    aliases: list[str] | None = None,
    max_unmatched_rate: float = 0.0,
    odds_type: str | None = None,
    drop_na_thres: float | None = None,
    target_event_status: str | None = None,
    target_event_time: str | None = None,
    input_event_status: str | None = None,
    input_event_time: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Download the training data and save the dataloader to a file.

    This is the only tool that downloads the seasons, and `output` is where it writes what it got. Every other tool
    reads that file, so a metered odds feed is bought once rather than once per call. What such a feed charges is
    between whoever is asking and the vendor they buy from.

    The event arguments decide the moment a model bets at. Leave them alone for the usual case, which is betting before
    the match with the price that was on offer then.
    """
    selection = _selection(
        stats,
        odds,
        leagues,
        divisions,
        years,
        odds_key_env,
        odds_markets,
        odds_regions,
        odds_moments,
        aliases,
        max_unmatched_rate,
    )
    extraction = _extraction(
        odds_type,
        drop_na_thres,
        target_event_status,
        target_event_time,
        input_event_status,
        input_event_time,
    )
    result: dict[str, Any] = await _offload(_extract_train_data, selection, extraction, output)
    return result


@server.tool()
async def extract_exploration_data(
    stats: str,
    odds: str | None = None,
    leagues: list[str] | None = None,
    divisions: list[int] | None = None,
    years: list[int] | None = None,
    odds_key_env: str = DEFAULT_KEY_ENV,
    odds_markets: list[str] | None = None,
    odds_regions: list[str] | None = None,
    odds_moments: list[str] | None = None,
    aliases: list[str] | None = None,
    max_unmatched_rate: float = 0.0,
    odds_type: str | None = None,
    drop_na_thres: float | None = None,
    target_event_status: str | None = None,
    target_event_time: str | None = None,
    input_event_status: str | None = None,
    input_event_time: str | None = None,
) -> dict[str, Any]:
    """Return the features on their own, with no targets and no odds.

    Use it to look at a sport before choosing what to select or model, or when the source carries no odds and so has
    nothing to predict.
    """
    selection = _selection(
        stats,
        odds,
        leagues,
        divisions,
        years,
        odds_key_env,
        odds_markets,
        odds_regions,
        odds_moments,
        aliases,
        max_unmatched_rate,
    )
    extraction = _extraction(
        odds_type,
        drop_na_thres,
        target_event_status,
        target_event_time,
        input_event_status,
        input_event_time,
    )
    result: dict[str, Any] = await _offload(_extract_exploration_data, selection, extraction)
    return result


@server.tool()
async def extract_fixtures_data(dataloader: str) -> dict[str, Any]:
    """Return the games that have not been played yet, from a dataloader `extract_train_data` saved.

    They take the shape the training data took, since the dataloader remembers what it was told.
    """
    result: dict[str, Any] = await _offload(_extract_fixtures_data, dataloader)
    return result


@server.tool()
async def backtest(
    dataloader: str,
    model: str = 'odds-comparison',
    alpha: float = 0.05,
    betting_markets: list[str] | None = None,
    init_cash: float | None = None,
    stake: float | None = None,
    model_odds_types: list[str] | None = None,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 0,
) -> list[dict[str, Any]]:
    """Return the backtesting results of a betting model on a saved dataloader.

    A ready-made model is named. A scikit-learn one built in Python is named by where it lives, as in
    `models.py:BETTOR`, since no set of arguments can describe an estimator.
    """
    strategy = _strategy(model, alpha, betting_markets, init_cash, stake, model_odds_types)
    result: list[dict[str, Any]] = await _offload(_backtest, dataloader, strategy, cv, n_jobs, verbose)
    return result


@server.tool()
async def fit(
    dataloader: str,
    output: str,
    model: str = 'odds-comparison',
    alpha: float = 0.05,
    betting_markets: list[str] | None = None,
    init_cash: float | None = None,
    stake: float | None = None,
    model_odds_types: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a model on a saved dataloader and save it, so it is fitted once and reused.

    `bet` reads what this writes.
    """
    strategy = _strategy(model, alpha, betting_markets, init_cash, stake, model_odds_types)
    result: dict[str, Any] = await _offload(_fit, dataloader, strategy, output)
    return result


@server.tool()
async def bet(dataloader: str, bettor: str) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet.

    It reads the dataloader `extract_train_data` saved and the model `fit` saved, so it downloads nothing and fits
    nothing.
    """
    result: list[dict[str, Any]] = await _offload(_bet, dataloader, bettor)
    return result


def _venue(reference: str) -> BaseVenue:
    """Return the venue a reference names."""
    built = build_venue(reference)
    if not isinstance(built, BaseVenue):
        msg = f'`{reference}` is a browser session, which has no bets of its own to place. Use the browser tools.'
        raise TypeError(msg)
    return built


def _intents(key: str, records: list[dict[str, Any]]) -> list[PlacementIntent]:
    """Return the bets a caller means to place."""
    return [
        PlacementIntent(
            identity=BetIdentity(key, record['match'], record['market'], record['selection']),
            stake=float(record['stake']),
            min_price=float(record['min_price']),
            value_bet=str(record.get('value_bet', '')),
        )
        for record in records
    ]


def _quote_records(quoted: PlacementQuote) -> dict[str, Any]:
    """Return a quote an agent can read and pass back."""
    return {
        'total_stake': quoted.total_stake,
        'total_exposure': quoted.total_exposure,
        'quoted_at': quoted.quoted_at.isoformat(),
        'intents': [
            {
                'match': intent.identity.match,
                'market': intent.identity.market,
                'selection': intent.identity.selection,
                'stake': intent.stake,
                'min_price': intent.min_price,
                'value_bet': intent.value_bet,
                'ref': intent.identity.ref,
            }
            for intent in quoted.intents
        ],
    }


def _quote_of(key: str, held: dict[str, Any]) -> PlacementQuote:
    """Return the quote a caller passed back."""
    return PlacementQuote(
        intents=_intents(key, held['intents']),
        total_stake=float(held['total_stake']),
        total_exposure=float(held['total_exposure']),
        quoted_at=pd.Timestamp(held['quoted_at']).to_pydatetime(),
    )


@server.tool()
async def execution_venue_info(venue: str) -> dict[str, Any]:
    """Return what a venue is and what its owner wrote down about the site.

    It answers for a venue with an API and for a bookmaker's website alike, since it is the tool that hands a website's
    notes to the agent. The notes come back exactly as they were written. The library does not read them: they are for
    you.
    """
    built = build_venue(venue)
    return {
        'key': built.key,
        'can_cancel': getattr(built, 'can_cancel', False),
        'url': getattr(built, 'url', None),
        'notes': getattr(built, 'notes', None),
    }


@server.tool()
async def execution_authenticate(venue: str) -> dict[str, Any]:
    """Authenticate at a venue, reading each secret from the variable the venue names."""
    built = _venue(venue)
    await built.authenticate()
    return {'venue': built.key, 'authenticated': True}


@server.tool()
async def execution_read_balance(venue: str) -> dict[str, Any]:
    """Return the balance and what is currently at stake."""
    built = _venue(venue)
    await built.authenticate()
    balance, exposure = await built.read_balance()
    return {'balance': balance, 'exposure': exposure}


@server.tool()
async def execution_list_markets(venue: str, matches: list[str]) -> list[dict[str, Any]]:
    """Return the markets a venue offers on the given matches, with their prices."""
    built = _venue(venue)
    await built.authenticate()
    return _records(await built.list_markets(matches))


def _fixtures(dataloader: str, bettor: str) -> tuple[Any, Any, Any]:
    """Return the fitted model and the upcoming matches it bets on."""
    loader, _ = load_dataloader(dataloader)
    fitted = load_bettor(bettor)
    X_fix, _, O_fix = loader.extract_fixtures_data()
    return fitted, X_fix, O_fix


@server.tool()
async def execution_quote(venue: str, dataloader: str, bettor: str, stake: float) -> dict[str, Any]:
    """Return what would be staked on the upcoming matches, before anything is.

    Pass `total_stake` and `total_exposure` back to `execution_place` to place the bets. Nothing is staked until you do.

    It reads the dataloader `extract_train_data` saved and the model `fit` saved, so it downloads nothing and fits
    nothing.
    """
    built = _venue(venue)
    await built.authenticate()
    fitted, X_fix, O_fix = await _offload(_fixtures, dataloader, bettor)
    if X_fix.empty or O_fix is None or O_fix.empty:
        return {'total_stake': 0.0, 'total_exposure': 0.0, 'quoted_at': None, 'intents': []}
    intents = value_bet_intents(built.key, fitted, X_fix, O_fix, stake)
    quoted = await run_quote(built, intents, ExposureLimits())
    return _quote_records(quoted)


@server.tool()
async def execution_place(
    venue: str,
    quote: dict[str, Any],
    confirm_stake: float | None = None,
    confirm_exposure: float | None = None,
    max_stake: float = 0.0,
    max_exposure: float = 0.0,
    kill: bool = False,
) -> list[dict[str, Any]]:
    """Place a quoted batch, staking nothing unless the quoted figures are passed back exactly.

    `confirm_stake` and `confirm_exposure` are the figures `execution_quote` returned. Anything else stakes nothing and
    says what the figures really are.
    """
    built = _venue(venue)
    await built.authenticate()
    limits = ExposureLimits(max_stake_per_bet=max_stake, max_total_exposure=max_exposure, killed=kill)
    receipts = await run_place(built, _quote_of(built.key, quote), limits, confirm_stake, confirm_exposure)
    return _records(receipts)


@server.tool()
async def execution_read_status(venue: str, intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return what the venue holds for these bets."""
    built = _venue(venue)
    await built.authenticate()
    identities = [intent.identity for intent in _intents(built.key, intents)]
    return _records(await built.read_status(identities))


@server.tool()
async def execution_cancel(venue: str, match: str, market: str, selection: str) -> dict[str, Any]:
    """Cancel a bet, where the venue cancels."""
    built = _venue(venue)
    await built.authenticate()
    receipt = await built.cancel(BetIdentity(built.key, match, market, selection))
    return {'status': receipt.status.value, 'detail': receipt.detail}


_SESSIONS: dict[str, BrowserSession] = {}


async def _session(venue: str) -> BrowserSession:
    """Return the browser session a reference names, opening it once and keeping it open.

    A login has to last across calls, and the browser is what holds it, so the session is kept here for as long as the
    server runs rather than opened and closed around each call.
    """
    if venue not in _SESSIONS:
        built = build_venue(venue)
        if isinstance(built, BaseVenue):
            msg = f'`{venue}` is a venue with an API, so it is placed at with `execution_place` rather than driven.'
            raise TypeError(msg)
        await built.start()
        _SESSIONS[venue] = built
    return _SESSIONS[venue]


@server.tool()
async def browser_navigate(venue: str, url: str) -> dict[str, Any]:
    """Go to a page of a bookmaker's website and return it.

    The page comes back as an accessibility snapshot: a ref for everything that can be acted on, which `browser_click`,
    `browser_type` and `browser_select` take. Driving a bookmaker's website breaches essentially every bookmaker's
    terms of service and risks the account being closed and the balance lost.
    """
    session = await _session(venue)
    shot = await session.navigate(url)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_snapshot(venue: str, selector: str | None = None, depth: int | None = None) -> dict[str, Any]:
    """Return the page, or a part of it.

    Read a part rather than the whole page where you can, since the whole page is the cost of every turn.
    """
    session = await _session(venue)
    shot = await session.snapshot(selector, depth)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_click(venue: str, ref: str) -> dict[str, Any]:
    """Click an element and return the page it produced.

    An element the site has disabled or hidden is not clicked and this says so, rather than reporting a click that did
    not happen.
    """
    session = await _session(venue)
    shot = await session.click(ref)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_type(venue: str, ref: str, text: str) -> dict[str, Any]:
    """Fill an element and return the page it produced."""
    session = await _session(venue)
    shot = await session.type(ref, text)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_select(venue: str, ref: str, value: str) -> dict[str, Any]:
    """Choose an option and return the page it produced."""
    session = await _session(venue)
    shot = await session.select(ref, value)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_fix(venue: str, match: str, locators: dict[str, str]) -> dict[str, Any]:
    """Pin what exploring found, so that placing does not have to find it again.

    Pin a role and an accessible name rather than a ref, since a ref belongs to one state of the page and is refused
    here. A price is not pinned at all: read it when the bet is placed.
    """
    session = await _session(venue)
    pinned = session.fix(match, locators)
    return {'match': pinned.match, 'url': pinned.url, 'locators': pinned.locators}


def _run(
    venue: BaseVenue,
    dataloader: str,
    bettor: str,
    stake: float,
    max_stake: float,
    max_exposure: float,
    confirm_total: float | None,
    window: str | None,
    seed: int,
) -> pd.DataFrame:
    """Place the value bets of the upcoming matches, one match at a time."""
    loader, _ = load_dataloader(dataloader)
    fitted = load_bettor(bettor)
    return asyncio.run(
        run_execute(
            venue,
            loader,
            fitted,
            stake=stake,
            max_stake=max_stake,
            max_exposure=max_exposure,
            confirm_total=confirm_total,
            window=pd.Timedelta(window) if window else None,
            seed=seed,
        ),
    )


@server.tool()
async def execution_run(
    venue: str,
    dataloader: str,
    bettor: str,
    stake: float,
    confirm_total: float | None = None,
    max_stake: float = 0.0,
    max_exposure: float = 0.0,
    window: str | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Place the value bets of the upcoming matches, one match at a time.

    It reads the dataloader `extract_train_data` saved and the model `fit` saved, keeps the matches the model bets on
    and can still reach, and places them in turn, waiting until each match's moment. Nothing stakes until
    `confirm_total` matches the total it quotes. Set `window` as `2h` for a live model, to bound how long it runs.
    """
    built = _venue(venue)
    receipts = await _offload(
        _run, built, dataloader, bettor, stake, max_stake, max_exposure, confirm_total, window, seed,
    )
    return _records(receipts)


def run() -> None:
    """Run the server."""
    server.run()
