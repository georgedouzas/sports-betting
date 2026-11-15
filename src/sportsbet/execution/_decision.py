"""Decide the one bet a model backs on an event."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import re

import pandas as pd

from ..evaluation import BaseBettor, find_latest_odds_column
from ._base import BetIdentity, PlacementIntent

LINE_PATTERN = re.compile(r'^(over|under)_(\d+(?:\.\d+)?)$')


def _map_selection(market: str, home_team: str, away_team: str) -> str | None:
    """Return the side a market backs, or `None` for a market the run does not know."""
    sides = {'home_win': home_team, 'away_win': away_team, 'draw': 'Draw'}
    if market in sides:
        return sides[market]
    line = LINE_PATTERN.match(market)
    if line is not None:
        return f'{line.group(1).capitalize()} {line.group(2)}'
    return None


def _read_source_price(odds: pd.DataFrame, market: str) -> float | None:
    """Return the latest price the odds carry for a market, or `None`."""
    column = find_latest_odds_column(list(odds.columns), market)
    if column is None:
        return None
    price = odds.iloc[0][column]
    return None if pd.isna(price) else float(price)


def decide_bet(
    event: str,
    bettor: BaseBettor,
    X: pd.DataFrame,
    O: pd.DataFrame,
    *,
    venue_key: str,
    stake: float,
) -> tuple[PlacementIntent | None, str]:
    """Return the one bet a fitted bettor backs on an event, and why.

    The bettor is applied to the event's data. When it backs several markets, the first in its market order is taken.
    The market is mapped to the side it backs: `home_win` to the home team, `away_win` to the away team, `draw` to
    `Draw`, and an over or under market to its line, as in `Over 2.5`. The lowest acceptable price is the source's
    latest price for that market, which is the price the model saw value at.

    Args:
        event:
            The event, as `Home vs Away`.
        bettor:
            A fitted bettor.
        X:
            The event's one-row features.
        O:
            The event's one-row odds.
        venue_key:
            The venue the bet goes on.
        stake:
            The fixed amount to stake.

    Returns:
        decision:
            The intended bet, or `None` when there is none, and a sentence saying what was decided.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sportsbet.execution import decide_bet
        >>> class Bettor:
        ...     betting_markets_ = np.array(['home_win', 'away_win'])
        ...     def bet(self, X, O):
        ...         return np.array([[False, True]])
        >>> X = pd.DataFrame({'home_team': ['Arsenal'], 'away_team': ['Chelsea']})
        >>> O = pd.DataFrame({'acme__home_win__preplay__0min': [1.8], 'acme__away_win__preplay__0min': [4.2]})
        >>> intent, reason = decide_bet('Arsenal vs Chelsea', Bettor(), X, O, venue_key='book', stake=10.0)
        >>> intent.identity.selection, intent.min_price
        ('Chelsea', 4.2)
        >>> reason
        'Backs Chelsea on away_win at 4.2 or better.'
    """
    markets = [str(market) for market in bettor.betting_markets_]
    value_bets = pd.DataFrame(bettor.bet(X, O), columns=markets)
    backed = [market for market in markets if bool(value_bets.iloc[0][market])]
    if not backed:
        return None, f'No value bet was found on {event}.'
    market, *others = backed
    not_placed = f' Not placed: {", ".join(others)}.' if others else ''
    selection = _map_selection(market, str(X.iloc[0]['home_team']), str(X.iloc[0]['away_team']))
    if selection is None:
        return None, f'`{market}` is not a market a run knows how to back.{not_placed}'
    price = _read_source_price(O, market)
    if price is None:
        return None, f'The source has no price for {market}, so no price was available to decide on.{not_placed}'
    intent = PlacementIntent(
        identity=BetIdentity(venue_key, event, market, selection),
        stake=stake,
        min_price=price,
        value_bet=f'{event}|{market}',
    )
    return intent, f'Backs {selection} on {market} at {price} or better.{not_placed}'
