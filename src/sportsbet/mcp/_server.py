"""Implements the server that lets an assistant drive the library.

The assistant lives outside the library. Nothing here calls a model, holds a model's key, or chooses one: the library
stays a set of estimators that behave the same way every time they are run, and the assistant is one more consumer of
it.

Every tool is handed the path of the same configuration the command line reads, rather than a description of a
dataloader in the arguments. One configuration and one reader means the two surfaces cannot come to mean different
things, and it keeps a key out of a tool call, where it would be written into a transcript.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
from mcp.server.fastmcp import FastMCP

from .._config import read_bettor, read_dataloader, read_module
from ..evaluation import backtest as run_backtest

server: FastMCP = FastMCP('sportsbet')

Answer = TypeVar('Answer')


async def _offload(work: Callable[..., Answer], *args: object) -> Answer:
    """Run the library in a thread, since it fetches with an event loop of its own.

    A tool is answered inside an event loop, and the library opens one to fetch. A loop cannot be opened inside a loop,
    so anything that might fetch is handed to a thread that has none.
    """
    return await asyncio.to_thread(work, *args)


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Return a frame as records, since an assistant cannot read a dataframe."""
    if frame is None or frame.empty:
        return []
    return [
        {col: (None if pd.isna(value) else value) for col, value in row.items()}
        for row in frame.astype(object).to_dict(orient='records')
    ]


def _report(config_path: str) -> tuple[Any, dict[str, Any]]:
    """Return the dataloader and what a preparation would fetch and cost, without fetching anything metered."""
    dataloader = read_dataloader(read_module(config_path))
    report = dataloader.prepare(dry_run=True)
    estimate = {
        'to_fetch': len(report.to_fetch),
        'held': len(report.held),
        'cost': dict(report.estimated_cost),
        'total_cost': sum(report.estimated_cost.values()),
        'unavailable': report.unavailable,
    }
    return dataloader, estimate


def _available_params(config_path: str) -> list[dict]:
    """Return the leagues, divisions and seasons that can be selected.

    Free, and downloads no data.
    """
    dataloader = read_dataloader(read_module(config_path))
    stats_source, *_ = dataloader.sources
    return stats_source.available_params()


def _estimate_preparation(config_path: str) -> dict[str, Any]:
    """Return what a preparation would fetch and exactly what it would cost, spending nothing.

    Call this before `prepare`. The cost is the real list of what would be bought, not a guess, and it is free to ask.
    """
    _, estimate = _report(config_path)
    return estimate


def _prepare(config_path: str, confirm_cost: int | None = None) -> dict[str, Any]:
    """Download the data a dataloader needs.

    A source that charges is not allowed to charge by surprise. If the preparation costs anything, `confirm_cost` must
    be the total that `estimate_preparation` reports, so that whoever is paying has been told the price first. A
    preparation that costs nothing needs no confirmation.
    """
    dataloader, estimate = _report(config_path)
    total = estimate['total_cost']
    if total and confirm_cost != total:
        msg = (
            f'This preparation costs {total}, and it has not been confirmed. Tell whoever is paying what it costs, '
            f'then call this again with `confirm_cost={total}`.'
        )
        raise ValueError(msg)
    report = dataloader.prepare()
    return {'fetched': len(report.to_fetch), 'held': len(report.held), 'cost': dict(report.estimated_cost)}


def _extract_train_data(config_path: str, odds_type: str | None = None) -> dict[str, Any]:
    """Return the training data of the prepared data."""
    dataloader = read_dataloader(read_module(config_path))
    X, Y, O = dataloader.extract_train_data(odds_type=odds_type)
    return {'X': _records(X), 'Y': _records(Y), 'O': _records(O)}


def _extract_fixtures_data(config_path: str, odds_type: str | None = None) -> dict[str, Any]:
    """Return the games that have not been played yet.

    The fixtures take the shape of the training data, so the training data is extracted first, with the same odds. A
    fixture that carried a different kind of price from the one a model learned on would be a column the model has never
    seen.

    A tool is answered on its own and remembers nothing between calls, so what a dataloader would have kept from an
    earlier extraction has to be established here.
    """
    dataloader = read_dataloader(read_module(config_path))
    dataloader.extract_train_data(odds_type=odds_type)
    X, _, O = dataloader.extract_fixtures_data()
    return {'X': _records(X), 'O': _records(O)}


def _backtest(config_path: str, odds_type: str | None = None) -> list[dict[str, Any]]:
    """Return the backtesting results of the configuration's bettor."""
    mod = read_module(config_path)
    dataloader, bettor = read_dataloader(mod), read_bettor(mod)
    X, Y, O = dataloader.extract_train_data(odds_type=odds_type)
    return _records(run_backtest(bettor, X, Y, O))


def _bet(config_path: str, odds_type: str | None = None) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
    mod = read_module(config_path)
    dataloader, bettor = read_dataloader(mod), read_bettor(mod)
    X, Y, O = dataloader.extract_train_data(odds_type=odds_type)
    bettor.fit(X, Y, O)
    X_fix, _, O_fix = dataloader.extract_fixtures_data()
    value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=list(bettor.betting_markets_))
    games = X_fix[['home_team', 'away_team']].reset_index()
    return _records(pd.concat([games, value_bets], axis=1))


@server.tool()
async def available_params(config_path: str) -> list[dict]:
    """Return the leagues, divisions and seasons that can be selected.

    Free, and downloads no data.
    """
    result: list[dict] = await _offload(_available_params, config_path)
    return result


@server.tool()
async def estimate_preparation(config_path: str) -> dict[str, Any]:
    """Return what a preparation would fetch and exactly what it would cost, spending nothing.

    Call this before `prepare`. The cost is the real list of what would be bought rather than a guess, and asking is
    free.
    """
    result: dict[str, Any] = await _offload(_estimate_preparation, config_path)
    return result


@server.tool()
async def prepare(config_path: str, confirm_cost: int | None = None) -> dict[str, Any]:
    """Download the data a dataloader needs.

    A source that charges is not allowed to charge by surprise. If the preparation costs anything, `confirm_cost` must
    be the total that `estimate_preparation` reports, so whoever is paying has been told the price first. A preparation
    that costs nothing needs no confirmation.
    """
    result: dict[str, Any] = await _offload(_prepare, config_path, confirm_cost)
    return result


@server.tool()
async def extract_train_data(config_path: str, odds_type: str | None = None) -> dict[str, Any]:
    """Return the training data of the prepared data."""
    result: dict[str, Any] = await _offload(_extract_train_data, config_path, odds_type)
    return result


@server.tool()
async def extract_fixtures_data(config_path: str, odds_type: str | None = None) -> dict[str, Any]:
    """Return the games that have not been played yet."""
    result: dict[str, Any] = await _offload(_extract_fixtures_data, config_path, odds_type)
    return result


@server.tool()
async def backtest(config_path: str, odds_type: str | None = None) -> list[dict[str, Any]]:
    """Return the backtesting results of the configuration's bettor."""
    result: list[dict[str, Any]] = await _offload(_backtest, config_path, odds_type)
    return result


@server.tool()
async def bet(config_path: str, odds_type: str | None = None) -> list[dict[str, Any]]:
    """Return the value bets of the games that have not been played yet."""
    result: list[dict[str, Any]] = await _offload(_bet, config_path, odds_type)
    return result


def run() -> None:
    """Run the server."""
    server.run()
