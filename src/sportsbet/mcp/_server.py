"""Serve the library's capabilities as MCP tools."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from mcp.server.fastmcp import FastMCP
from sklearn.model_selection import TimeSeriesSplit

from ..dataloaders import DEFAULT_KEY_ENV, build_dataloader, build_extraction_settings, load_dataloader
from ..evaluation import backtest as run_backtest
from ..evaluation import build_bettor, load_bettor, save_bettor
from ..execution import (
    BaseVenue,
    BetIdentity,
    BrowserSession,
    ExecutionError,
    PlacementIntent,
    build_venue,
    execute_event,
)

server: FastMCP = FastMCP('sportsbet')

Answer = TypeVar('Answer')
Selection = dict[str, Any]
_SESSIONS: dict[str, BrowserSession] = {}


async def _offload(work: Callable[..., Answer], *args: object) -> Answer:
    """Run the work in a thread, off the tool's event loop."""
    return await asyncio.to_thread(work, *args)


def _to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Return a frame as records."""
    if frame is None or frame.empty:
        return []
    return [
        {col: (None if pd.isna(value) else value) for col, value in row.items()}
        for row in frame.astype(object).to_dict(orient='records')
    ]


def _build_selection(
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
    }


def _build_extraction(
    odds_type: str | None,
    drop_na_thres: float | None,
    target_event_status: str | None,
    target_event_time: str | None,
    input_event_status: str | None,
    input_event_time: str | None,
) -> dict[str, Any]:
    """Return how a tool was told to extract."""
    return build_extraction_settings(
        odds_type=odds_type,
        drop_na_thres=drop_na_thres,
        target_event_status=target_event_status,
        target_event_time=target_event_time,
        input_event_status=input_event_status,
        input_event_time=input_event_time,
    )


def _read_available_params(selection: Selection) -> list[dict]:
    """Return what can be selected."""
    stats_source, *_ = build_dataloader(**selection).sources_
    return stats_source.list_available_params()


def _read_odds_types(selection: Selection) -> list[str]:
    """Return the odds types a selection carries."""
    return list(build_dataloader(**selection).get_odds_types())


def _extract_train_data(selection: Selection, extraction: dict[str, Any], output: str | None) -> dict[str, Any]:
    """Download the training data, and write the dataloader where the other tools can read it."""
    dataloader = build_dataloader(**selection)
    X, Y, O = dataloader.extract_train_data(**extraction)
    if output is not None:
        dataloader.save(output)
    return {'X': _to_records(X), 'Y': _to_records(Y), 'O': _to_records(O), 'output': output}


def _extract_exploration_data(selection: Selection, extraction: dict[str, Any]) -> dict[str, Any]:
    """Return the features on their own, with no targets and no odds."""
    settings = {name: value for name, value in extraction.items() if name != 'odds_type'}
    X = build_dataloader(**selection).extract_exploration_data(**settings)
    return {'X': _to_records(X)}


def _extract_fixtures_data(dataloader: str) -> dict[str, Any]:
    """Return the games that have not been played yet, from a saved dataloader."""
    loader = load_dataloader(dataloader)
    X, _, O = loader.extract_fixtures_data()
    return {'X': _to_records(X), 'O': _to_records(O)}


def _backtest(dataloader: str, model: str, cv: int, n_jobs: int, verbose: int) -> list[dict[str, Any]]:
    """Return the backtesting results of a model on a saved dataloader."""
    X, Y, O = load_dataloader(dataloader).extract_train_data()
    bettor = build_bettor(model)
    return _to_records(run_backtest(bettor, X, Y, O, cv=TimeSeriesSplit(cv), n_jobs=n_jobs, verbose=verbose))


def _fit(dataloader: str, model: str, output: str) -> dict[str, Any]:
    """Fit a model on a saved dataloader and save it."""
    X, Y, O = load_dataloader(dataloader).extract_train_data()
    bettor = build_bettor(model)
    bettor.fit(X, Y, O)
    save_bettor(bettor, output)
    return {'output': output, 'betting_markets': list(bettor.betting_markets_)}


def _bet(dataloader: str, bettor: str) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
    loader = load_dataloader(dataloader)
    fitted = load_bettor(bettor)
    X_fix, _, O_fix = loader.extract_fixtures_data()
    if X_fix.empty or O_fix is None or O_fix.empty:
        return []
    value_bets = pd.DataFrame(fitted.bet(X_fix, O_fix), columns=list(fitted.betting_markets_))
    games = X_fix[['home_team', 'away_team']].reset_index()
    return _to_records(pd.concat([games, value_bets], axis=1))


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
) -> list[dict]:
    """Return the leagues, divisions and seasons that can be selected."""
    selection = _build_selection(
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
    )
    result: list[dict] = await _offload(_read_available_params, selection)
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
) -> list[str]:
    """Return the odds types a selection carries."""
    selection = _build_selection(
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
    )
    result: list[str] = await _offload(_read_odds_types, selection)
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
    odds_type: str | None = None,
    drop_na_thres: float | None = None,
    target_event_status: str | None = None,
    target_event_time: str | None = None,
    input_event_status: str | None = None,
    input_event_time: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Download the training data and save the dataloader to a file.

    The event arguments decide the moment a model bets at, and the default is before the match.
    """
    selection = _build_selection(
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
    )
    extraction = _build_extraction(
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
    odds_type: str | None = None,
    drop_na_thres: float | None = None,
    target_event_status: str | None = None,
    target_event_time: str | None = None,
    input_event_status: str | None = None,
    input_event_time: str | None = None,
) -> dict[str, Any]:
    """Return the features on their own."""
    selection = _build_selection(
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
    )
    extraction = _build_extraction(
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

    They take the shape the training data took.
    """
    result: dict[str, Any] = await _offload(_extract_fixtures_data, dataloader)
    return result


@server.tool()
async def backtest(
    dataloader: str,
    model: str,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 0,
) -> list[dict[str, Any]]:
    """Return the backtesting results of a betting model on a saved dataloader.

    The model is a scikit-learn estimator written as a Python expression, as in `OddsComparisonBettor(alpha=0.05)`, or
    one built in a file, named by where it lives, as in `models.py:BETTOR`.
    """
    result: list[dict[str, Any]] = await _offload(_backtest, dataloader, model, cv, n_jobs, verbose)
    return result


@server.tool()
async def fit(dataloader: str, output: str, model: str) -> dict[str, Any]:
    """Fit a model on a saved dataloader and save it.

    The model is a scikit-learn estimator written as a Python expression, as in `OddsComparisonBettor(alpha=0.05)`, or
    one built in a file, named by where it lives, as in `models.py:BETTOR`.
    """
    result: dict[str, Any] = await _offload(_fit, dataloader, model, output)
    return result


@server.tool()
async def bet(dataloader: str, bettor: str) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet.

    It reads the dataloader `extract_train_data` saved and the model `fit` saved.
    """
    result: list[dict[str, Any]] = await _offload(_bet, dataloader, bettor)
    return result


def _load_venue(reference: str) -> BaseVenue:
    """Return the venue a reference names."""
    built = build_venue(reference)
    if not isinstance(built, BaseVenue):
        msg = f'`{reference}` is a browser session, which has no bets of its own to place. Use the browser tools.'
        raise ExecutionError(msg)
    return built


def _build_intents(key: str, records: list[dict[str, Any]]) -> list[PlacementIntent]:
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


def _run_event(
    venue: str,
    dataloader: str,
    bettor: str,
    event: str,
    stake: float,
    urls: list[str],
    live: bool,
    poll: str,
    output: str | None,
) -> pd.DataFrame:
    """Watch one event and place the model's bet at its moment."""
    session = build_venue(venue)
    if isinstance(session, BaseVenue):
        msg = f'`{venue}` is a venue with an API rather than a browser session, so `execution_run` does not apply.'
        raise ExecutionError(msg)
    loader = load_dataloader(dataloader)
    fitted = load_bettor(bettor)
    receipts = asyncio.run(
        execute_event(event, fitted, loader, session, stake=stake, urls=urls, live=live, poll=pd.Timedelta(poll)),
    )
    if output is not None:
        written = Path(output) / 'sports-betting-data'
        written.mkdir(parents=True, exist_ok=True)
        receipts.to_csv(written / 'receipts.csv', index=False)
    return receipts


@server.tool()
async def execution_venue_info(venue: str) -> dict[str, Any]:
    """Return what a venue is and what its owner wrote down about the site."""
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
    built = _load_venue(venue)
    await built.authenticate()
    return {'venue': built.key, 'authenticated': True}


@server.tool()
async def execution_read_balance(venue: str) -> dict[str, Any]:
    """Return the balance and what is currently at stake."""
    built = _load_venue(venue)
    await built.authenticate()
    balance, exposure = await built.read_balance()
    return {'balance': balance, 'exposure': exposure}


@server.tool()
async def execution_list_markets(venue: str, matches: list[str]) -> list[dict[str, Any]]:
    """Return the markets a venue offers on the given matches, with their prices."""
    built = _load_venue(venue)
    await built.authenticate()
    return _to_records(await built.list_markets(matches))


@server.tool()
async def execution_read_status(venue: str, intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return what the venue holds for these bets."""
    built = _load_venue(venue)
    await built.authenticate()
    identities = [intent.identity for intent in _build_intents(built.key, intents)]
    return _to_records(await built.read_status(identities))


@server.tool()
async def execution_cancel(venue: str, match: str, market: str, selection: str) -> dict[str, Any]:
    """Cancel a bet, where the venue cancels."""
    built = _load_venue(venue)
    await built.authenticate()
    receipt = await built.cancel(BetIdentity(built.key, match, market, selection))
    return {'status': receipt.status.value, 'detail': receipt.detail}


@server.tool()
async def execution_run(
    venue: str,
    dataloader: str,
    bettor: str,
    event: str,
    stake: float,
    urls: list[str] | None = None,
    live: bool = False,
    poll: str = '30s',
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Watch one event and place the model's bet at its moment.

    It reads the dataloader `extract_train_data` saved and the model `fit` saved, explores the URLs to match the event,
    ensures the browser session is logged in, monitors the event to its fitted moment, and places the stake on the
    model's selection when the model finds value. Without `live` it stakes nothing.
    """
    receipts = await _offload(_run_event, venue, dataloader, bettor, event, stake, urls or [], live, poll, output)
    return _to_records(receipts)


async def _open_session(venue: str) -> BrowserSession:
    """Return the browser session a reference names, opening it once and keeping it open."""
    if venue not in _SESSIONS:
        built = build_venue(venue)
        if isinstance(built, BaseVenue):
            msg = f'`{venue}` is a venue with an API rather than a browser session, so the browser tools do not apply.'
            raise ExecutionError(msg)
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
    session = await _open_session(venue)
    shot = await session.navigate(url)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_snapshot(venue: str, selector: str | None = None, depth: int | None = None) -> dict[str, Any]:
    """Return the page, or a part of it."""
    session = await _open_session(venue)
    shot = await session.read_snapshot(selector, depth)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_click(venue: str, ref: str) -> dict[str, Any]:
    """Click an element and return the page it produced."""
    session = await _open_session(venue)
    shot = await session.click(ref)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_type(venue: str, ref: str, text: str) -> dict[str, Any]:
    """Fill an element and return the page it produced."""
    session = await _open_session(venue)
    shot = await session.type(ref, text)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_select(venue: str, ref: str, value: str) -> dict[str, Any]:
    """Choose an option and return the page it produced."""
    session = await _open_session(venue)
    shot = await session.select(ref, value)
    return {'yaml': shot.yaml, 'url': shot.url}


@server.tool()
async def browser_fix(venue: str, match: str, locators: dict[str, str]) -> dict[str, Any]:
    """Pin the locators exploring found."""
    session = await _open_session(venue)
    pinned = session.fix(match, locators)
    return {'match': pinned.match, 'url': pinned.url, 'locators': pinned.locators}


def run() -> None:
    """Run the server."""
    server.run()
