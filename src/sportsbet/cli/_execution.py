"""Module that contains the execution commands of the CLI.

There is no dry run flag. A dry run is what happens when the quoted figures are not passed back, so there is no default
to get wrong and no flag to forget. `place` without them prints the quote, stakes nothing and exits non-zero.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from .._selection import build_venue
from ..evaluation import load_bettor
from ..execution import BaseVenue, BetIdentity, ExposureLimits, PlacementIntent, PlacementQuote
from ..execution import place as run_place
from ..execution import quote as run_quote
from ._utils import load_dataloader, print_console, reported


def _venue(venue_ref: str) -> BaseVenue:
    """Return the venue a reference names, authenticated."""
    built = build_venue(venue_ref)
    asyncio.run(built.authenticate())
    return built


def _limits(max_stake: float, max_exposure: float, kill: bool) -> ExposureLimits:
    """Return the ceilings a placement answers to."""
    return ExposureLimits(max_stake_per_bet=max_stake, max_total_exposure=max_exposure, killed=kill)


def _write_quote(quoted: PlacementQuote, path: str) -> None:
    """Write a quote where `place` can read it."""
    Path(path).write_text(
        json.dumps(
            {
                'total_stake': quoted.total_stake,
                'total_exposure': quoted.total_exposure,
                'quoted_at': quoted.quoted_at.isoformat(),
                'intents': [
                    {
                        'venue': intent.identity.venue,
                        'match': intent.identity.match,
                        'market': intent.identity.market,
                        'selection': intent.identity.selection,
                        'stake': intent.stake,
                        'min_price': intent.min_price,
                        'value_bet': intent.value_bet,
                    }
                    for intent in quoted.intents
                ],
            },
            indent=2,
        ),
    )


def _read_quote(path: str) -> PlacementQuote:
    """Return the quote a file holds."""
    held = json.loads(Path(path).read_text())
    return PlacementQuote(
        intents=[
            PlacementIntent(
                identity=BetIdentity(intent['venue'], intent['match'], intent['market'], intent['selection']),
                stake=intent['stake'],
                min_price=intent['min_price'],
                value_bet=intent['value_bet'],
            )
            for intent in held['intents']
        ],
        total_stake=held['total_stake'],
        total_exposure=held['total_exposure'],
        quoted_at=pd.Timestamp(held['quoted_at']).to_pydatetime(),
    )


def _shown(quoted: PlacementQuote) -> pd.DataFrame:
    """Return a quote as a table."""
    return pd.DataFrame.from_records(
        [
            {
                'match': intent.identity.match,
                'market': intent.identity.market,
                'selection': intent.identity.selection,
                'stake': intent.stake,
                'min_price': intent.min_price,
            }
            for intent in quoted.intents
        ],
    )


@click.group()
def execution() -> None:
    """Place the value bets a model found, at a venue where you hold an account.

    Nothing is staked until the figures `quote` returns are passed back to `place` exactly. Read the execution page of
    the user guide before using any of this: it spends real money.
    """
    return


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='A ready-made venue, or one of your own as `venue.py:VENUE`.')
def venue(venue_ref: str) -> None:
    """Show what a venue is and what it was told about the site."""
    with reported():
        built = build_venue(venue_ref)
        Console().print(
            Panel.fit(
                f'[bold]{built.key}[/bold]\ncancels: {built.can_cancel}\n\n{getattr(built, "notes", "") or ""}',
            ),
        )


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='A ready-made venue, or one of your own as `venue.py:VENUE`.')
@click.option(
    '--dataloader',
    '-d',
    'dataloader_path',
    required=True,
    type=click.Path(exists=True),
    help='A saved dataloader.',
)
def markets(venue_ref: str, dataloader_path: str) -> None:
    """Show the markets a venue offers on the upcoming matches, with their prices."""
    with reported():
        built = _venue(venue_ref)
        loader, _ = load_dataloader(dataloader_path)
        X_fix, _, _ = loader.extract_fixtures_data()
        matches = [f'{row.home_team} vs {row.away_team}' for row in X_fix.itertuples()]
        print_console([asyncio.run(built.list_markets(matches))], ['Markets'])


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='A ready-made venue, or one of your own as `venue.py:VENUE`.')
def balance(venue_ref: str) -> None:
    """Show the balance and what is currently at stake."""
    with reported():
        built = _venue(venue_ref)
        held, exposure = asyncio.run(built.read_balance())
        Console().print(Panel.fit(f'balance: [bold]{held}[/bold]\nexposure: [bold]{exposure}[/bold]'))


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='A ready-made venue, or one of your own as `venue.py:VENUE`.')
@click.option(
    '--dataloader',
    '-d',
    'dataloader_path',
    required=True,
    type=click.Path(exists=True),
    help='A saved dataloader.',
)
@click.option(
    '--bettor',
    '-b',
    'bettor_path',
    required=True,
    type=click.Path(exists=True),
    help='A model saved by `fit`.',
)
@click.option('--stake', type=float, required=True, help='What to stake on each value bet.')
@click.option('--output', '-o', 'output', required=True, type=click.Path(), help='Where to write the quote.')
def quote(venue_ref: str, dataloader_path: str, bettor_path: str, stake: float, output: str) -> None:
    """Show what would be staked on the upcoming matches, and write it for `place`."""
    with reported():
        built = _venue(venue_ref)
        loader, _ = load_dataloader(dataloader_path)
        bettor = load_bettor(bettor_path)
        X_fix, _, O_fix = loader.extract_fixtures_data()
        if X_fix.empty or O_fix is None or O_fix.empty:
            Console().print(Panel.fit('[bold red]There are no upcoming matches to bet on.'))
            return
        intents = _intents(built.key, bettor, X_fix, O_fix, stake)
        if not intents:
            Console().print(Panel.fit('[bold red]The model found no value bets.'))
            return
        quoted = asyncio.run(run_quote(built, intents, ExposureLimits()))
        print_console([_shown(quoted)], ['What would be staked'])
        Console().print(
            f'\nTotal stake [bold]{quoted.total_stake}[/bold], total exposure [bold]{quoted.total_exposure}[/bold].'
            f'\nTo place these bets, pass both back:'
            f'\n  --confirm-stake {quoted.total_stake} --confirm-exposure {quoted.total_exposure}',
        )
        _write_quote(quoted, output)


def _intents(key: str, bettor: object, X_fix: pd.DataFrame, O_fix: pd.DataFrame, stake: float) -> list[PlacementIntent]:
    """Return an intent for each value bet the model found."""
    markets_ = list(bettor.betting_markets_)  # type: ignore[attr-defined]
    value_bets = pd.DataFrame(bettor.bet(X_fix, O_fix), columns=markets_, index=X_fix.index)  # type: ignore[attr-defined]
    intents = []
    for position, (index, row) in enumerate(value_bets.iterrows()):
        game = X_fix.loc[index]
        match = f'{game["home_team"]} vs {game["away_team"]}'
        for market in markets_:
            if not row[market]:
                continue
            price = O_fix.iloc[position].get(f'{market}__odds')
            intents.append(
                PlacementIntent(
                    identity=BetIdentity(key, match, market, str(game['home_team'])),
                    stake=stake,
                    min_price=float(price) if price is not None and not pd.isna(price) else 1.01,
                    value_bet=f'{match}|{market}',
                ),
            )
    return intents


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='A ready-made venue, or one of your own as `venue.py:VENUE`.')
@click.option(
    '--quote',
    '-q',
    'quote_path',
    required=True,
    type=click.Path(exists=True),
    help='A quote written by `quote`.',
)
@click.option('--confirm-stake', type=float, help='The quoted stake, passed back to place the bets.')
@click.option('--confirm-exposure', type=float, help='The quoted exposure, passed back to place the bets.')
@click.option('--max-stake', type=float, default=0.0, help='The most to stake on one bet. Zero leaves it open.')
@click.option('--max-exposure', type=float, default=0.0, help='The most to have at stake at once. Zero leaves it open.')
@click.option('--kill', is_flag=True, help='Stop placing.')
@click.option('--output', '-o', 'data_path', type=click.Path(), help='A directory to write the receipts to, as CSV.')
def place(
    venue_ref: str,
    quote_path: str,
    confirm_stake: float | None,
    confirm_exposure: float | None,
    max_stake: float,
    max_exposure: float,
    kill: bool,
    data_path: str | None,
) -> None:
    """Place a quoted batch, staking nothing unless the quoted figures are passed back."""
    with reported():
        built = _venue(venue_ref)
        quoted = _read_quote(quote_path)
        receipts = asyncio.run(
            run_place(built, quoted, _limits(max_stake, max_exposure, kill), confirm_stake, confirm_exposure),
        )
        print_console([receipts], ['Receipts'])
        if data_path is not None:
            written = Path(data_path) / 'sports-betting-data'
            written.mkdir(parents=True, exist_ok=True)
            receipts.to_csv(written / 'receipts.csv', index=False)
        if not (receipts['stake'] > 0).any():
            raise SystemExit(1)
