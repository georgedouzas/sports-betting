"""Implements the server that lets an agent drive the library.

The agent lives outside the library. Nothing here calls a model, holds a model's key, or chooses one: the library stays
a set of estimators that behave the same way every time they are run, and the agent is one more consumer of it.

A tool is told what to do in its arguments, exactly as a command is, so there is no file to write first and nothing left
behind to fall out of date. A key is never one of those arguments: what is named is the environment variable holding it,
so the key itself stays out of a transcript.
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

from .._selection import DEFAULT_KEY_ENV, build_bettor, build_dataloader
from ..evaluation import backtest as run_backtest

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
    """Return a frame as records, since an agent cannot read a dataframe."""
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


def _extract_train_data(selection: Selection, odds_type: str | None) -> dict[str, Any]:
    """Return the training data."""
    X, Y, O = build_dataloader(**selection).extract_train_data(odds_type=odds_type)
    return {'X': _records(X), 'Y': _records(Y), 'O': _records(O)}


def _extract_fixtures_data(selection: Selection, odds_type: str | None) -> dict[str, Any]:
    """Return the games that have not been played yet.

    The training data is extracted first, so the fixtures take its shape and the same kind of odds. A tool is answered
    on its own and remembers nothing, so what a dataloader would have kept from an earlier extraction is established
    here.
    """
    dataloader = build_dataloader(**selection)
    dataloader.extract_train_data(odds_type=odds_type)
    X, _, O = dataloader.extract_fixtures_data()
    return {'X': _records(X), 'O': _records(O)}


def _backtest(
    selection: Selection,
    odds_type: str | None,
    strategy: Strategy,
    cv: int,
) -> list[dict[str, Any]]:
    """Return the backtesting results of a model."""
    dataloader, bettor = build_dataloader(**selection), build_bettor(**strategy)
    X, Y, O = dataloader.extract_train_data(odds_type=odds_type)
    return _records(run_backtest(bettor, X, Y, O, cv=TimeSeriesSplit(cv)))


def _bet(selection: Selection, odds_type: str | None, strategy: Strategy) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
    dataloader, bettor = build_dataloader(**selection), build_bettor(**strategy)
    X, Y, O = dataloader.extract_train_data(odds_type=odds_type)
    bettor.fit(X, Y, O)
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    if X_fix.empty:
        return []
    value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=list(bettor.betting_markets_))
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
) -> dict[str, Any]:
    """Return the training data.

    It downloads the selected seasons. What a metered odds feed charges for them is between whoever is asking and the
    vendor they buy from.
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
    result: dict[str, Any] = await _offload(_extract_train_data, selection, odds_type)
    return result


@server.tool()
async def extract_fixtures_data(
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
) -> dict[str, Any]:
    """Return the games that have not been played yet."""
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
    result: dict[str, Any] = await _offload(_extract_fixtures_data, selection, odds_type)
    return result


@server.tool()
async def backtest(
    stats: str,
    odds: str | None = None,
    model: str = 'odds-comparison',
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
    alpha: float = 0.05,
    betting_markets: list[str] | None = None,
    init_cash: float | None = None,
    stake: float | None = None,
    model_odds_types: list[str] | None = None,
    cv: int = 3,
) -> list[dict[str, Any]]:
    """Return the backtesting results of a betting model.

    A ready-made model is named. A scikit-learn one built in Python is named by where it lives, as in
    `models.py:BETTOR`, since no set of arguments can describe an estimator.
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
    strategy = _strategy(model, alpha, betting_markets, init_cash, stake, model_odds_types)
    result: list[dict[str, Any]] = await _offload(_backtest, selection, odds_type, strategy, cv)
    return result


@server.tool()
async def bet(
    stats: str,
    odds: str | None = None,
    model: str = 'odds-comparison',
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
    alpha: float = 0.05,
    betting_markets: list[str] | None = None,
    init_cash: float | None = None,
    stake: float | None = None,
    model_odds_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
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
    strategy = _strategy(model, alpha, betting_markets, init_cash, stake, model_odds_types)
    result: list[dict[str, Any]] = await _offload(_bet, selection, odds_type, strategy)
    return result


def run() -> None:
    """Run the server."""
    server.run()
