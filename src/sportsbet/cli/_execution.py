"""Run the execution commands from the command line."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from ..dataloaders import load_dataloader
from ..evaluation import load_bettor
from ..execution import (
    BaseVenue,
    BetIdentity,
    BrowserSession,
    FixedSession,
    build_venue,
    execute_event,
)
from ._building import _report_errors
from ._utils import _print_console


def _load_venue(venue_ref: str) -> BaseVenue:
    """Return the venue a reference names, authenticated."""
    built = build_venue(venue_ref)
    if not isinstance(built, BaseVenue):
        msg = f'`{venue_ref}` is a browser session, which has no bets of its own to place. Use `execution page`.'
        raise click.UsageError(msg)
    asyncio.run(built.authenticate())
    return built


def _load_session(venue_ref: str) -> BrowserSession:
    """Return the browser session a reference names."""
    built = build_venue(venue_ref)
    if isinstance(built, BaseVenue):
        msg = f'`{venue_ref}` is a venue with an API rather than a browser session, so `page` does not apply.'
        raise click.UsageError(msg)
    return built


@contextmanager
def _logging_to_terminal() -> Iterator[None]:
    """Show the run's log on the terminal while a command runs."""
    logger = logging.getLogger('sportsbet.execution')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@click.group()
def execution() -> None:
    """Place the value bets a model found, at a venue where you hold an account.

    Read the execution page of the user guide before using any of this: it spends real money.
    """
    return


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
def venue(venue_ref: str) -> None:
    """Show what a venue is and what it was told about the site."""
    with _report_errors():
        built = build_venue(venue_ref)
        cancels = getattr(built, 'can_cancel', False)
        Console().print(
            Panel.fit(f'[bold]{built.key}[/bold]\ncancels: {cancels}\n\n{getattr(built, "notes", "") or ""}'),
        )


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your browser session, as `venue.py:VENUE`.')
@click.option(
    '--dataloader',
    '-d',
    'dataloader_path',
    required=True,
    type=click.Path(exists=True),
    help='A saved dataloader configured for the event.',
)
@click.option(
    '--bettor',
    '-b',
    'bettor_path',
    required=True,
    type=click.Path(exists=True),
    help='A model saved by `fit`.',
)
@click.option('--event', required=True, help='The one event to act on, as `Home vs Away`.')
@click.option('--stake', type=float, required=True, help='The fixed stake to place.')
@click.option('--url', 'urls', multiple=True, help='A candidate bookmaker URL. Repeatable.')
@click.option('--live', is_flag=True, help='Arm the run. Off by default, which is a dry run.')
@click.option('--poll', default='30s', help='The source poll interval, as `30s` or `2min`.')
@click.option('--output', '-o', 'output', type=click.Path(), help='A directory to write the receipt CSV to.')
def run(
    venue_ref: str,
    dataloader_path: str,
    bettor_path: str,
    event: str,
    stake: float,
    urls: tuple[str, ...],
    live: bool,
    poll: str,
    output: str | None,
) -> None:
    """Watch one event and place the model's bet at its moment.

    It explores the URLs to match the event, prompts the login, logs the event to the terminal, and at the fitted moment
    places the stake on the model's selection. Without `--live` it stakes nothing and logs the bet it would have made.
    """
    with _report_errors(), _logging_to_terminal():
        session = _load_session(venue_ref)
        loader = load_dataloader(dataloader_path)
        bettor = load_bettor(bettor_path)
        receipts = asyncio.run(
            execute_event(
                event,
                bettor,
                loader,
                session,
                stake=stake,
                urls=list(urls),
                live=live,
                poll=pd.Timedelta(poll),
            ),
        )
        if not receipts.empty:
            _print_console([receipts], ['Receipts'])
            if output is not None:
                written = Path(output) / 'sports-betting-data'
                written.mkdir(parents=True, exist_ok=True)
                receipts.to_csv(written / 'receipts.csv', index=False)


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
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
    with _report_errors():
        built = _load_venue(venue_ref)
        loader = load_dataloader(dataloader_path)
        X_fix, _, _ = loader.extract_fixtures_data()
        matches = [f'{row.home_team} vs {row.away_team}' for row in X_fix.itertuples()]
        _print_console([asyncio.run(built.list_markets(matches))], ['Markets'])


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
def balance(venue_ref: str) -> None:
    """Show the balance and what is currently at stake."""
    with _report_errors():
        built = _load_venue(venue_ref)
        held, exposure = asyncio.run(built.read_balance())
        Console().print(Panel.fit(f'balance: [bold]{held}[/bold]\nexposure: [bold]{exposure}[/bold]'))


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
@click.option('--match', required=True, help='The match the bet is on.')
@click.option('--market', required=True, help='The market the bet is on.')
@click.option('--selection', required=True, help='The selection the bet backs.')
def status(venue_ref: str, match: str, market: str, selection: str) -> None:
    """Show what the venue holds for one bet."""
    with _report_errors():
        built = _load_venue(venue_ref)
        identity = BetIdentity(built.key, match, market, selection)
        _print_console([asyncio.run(built.read_status([identity]))], ['What the venue holds'])


@execution.command()
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
@click.option('--match', required=True, help='The match the bet is on.')
@click.option('--market', required=True, help='The market the bet is on.')
@click.option('--selection', required=True, help='The selection the bet backs.')
def cancel(venue_ref: str, match: str, market: str, selection: str) -> None:
    """Cancel a bet, where the venue cancels."""
    with _report_errors():
        built = _load_venue(venue_ref)
        receipt = asyncio.run(built.cancel(BetIdentity(built.key, match, market, selection)))
        Console().print(Panel.fit(receipt.detail))


@execution.group()
def page() -> None:
    """Read and act on a bookmaker's website.

    Each command is a whole session: it opens the browser, goes to the page, does the one thing and closes. A ref comes
    from the snapshot of the page it was read on, so pass the same `--url` that produced it.

    Driving a bookmaker's website breaches essentially every bookmaker's terms of service and risks the account being
    closed and the balance lost. The library supplies the browser and the page, and the knowledge of the site is yours.
    """
    return


def _parse_locators(given: tuple[str, ...]) -> dict[str, str]:
    """Return what was found, each of them a name and a locator."""
    found = {}
    for pair in given:
        name, sep, locator = pair.partition('=')
        if not sep or not name or not locator:
            msg = f'`{pair}` should be a name and a locator, as in `stake=textbox[name="Stake"]`.'
            raise click.UsageError(msg)
        found[name] = locator
    return found


async def _read(session: BrowserSession, url: str | None, selector: str | None, depth: int | None) -> str:
    """Open the browser, read the page and close it."""
    try:
        await session.navigate(url or session.url)
        shot = await session.read_snapshot(selector, depth)
        return shot.yaml
    finally:
        await session.stop()


async def _act(
    session: BrowserSession,
    url: str,
    click_ref: str | None,
    type_ref: str | None,
    text: str | None,
    select_ref: str | None,
    value: str | None,
) -> str:
    """Open the browser, act on the page and close it."""
    try:
        await session.navigate(url)
        if click_ref:
            shot = await session.click(click_ref)
        elif type_ref:
            shot = await session.type(type_ref, text or '')
        elif select_ref:
            shot = await session.select(select_ref, value or '')
        else:
            msg = 'Name what to do: `--click`, `--type` with `--text`, or `--select` with `--value`.'
            raise click.UsageError(msg)
        return shot.yaml
    finally:
        await session.stop()


async def _fix(session: BrowserSession, url: str, match: str, locators: dict[str, str]) -> FixedSession:
    """Open the browser, pin what was found and close it."""
    try:
        await session.navigate(url)
        return session.fix(match, locators)
    finally:
        await session.stop()


@page.command('read')
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
@click.option('--url', help='The page to read. Without it the venue\'s own url is read.')
@click.option('--selector', help='The part of the page to read. Reading a part keeps a turn cheap.')
@click.option('--depth', type=int, help='How far down to read.')
def page_read(venue_ref: str, url: str | None, selector: str | None, depth: int | None) -> None:
    """Show a page an agent can act on."""
    with _report_errors():
        session = _load_session(venue_ref)
        Console().print(asyncio.run(_read(session, url, selector, depth)))


@page.command('act')
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
@click.option('--url', required=True, help='The page the ref was read on.')
@click.option('--click', 'click_ref', help='The ref of an element to click.')
@click.option('--type', 'type_ref', help='The ref of an element to fill.')
@click.option('--text', help='What to fill it with.')
@click.option('--select', 'select_ref', help='The ref of an element to choose an option in.')
@click.option('--value', help='The option to choose.')
def page_act(
    venue_ref: str,
    url: str,
    click_ref: str | None,
    type_ref: str | None,
    text: str | None,
    select_ref: str | None,
    value: str | None,
) -> None:
    """Act on a page and show what the action produced."""
    with _report_errors():
        session = _load_session(venue_ref)
        Console().print(asyncio.run(_act(session, url, click_ref, type_ref, text, select_ref, value)))


@page.command('fix')
@click.option('--venue', 'venue_ref', required=True, help='Your venue, as `venue.py:VENUE`.')
@click.option('--url', required=True, help='The page that was explored.')
@click.option('--match', required=True, help='The match to pin the session to.')
@click.option(
    '--locator',
    'locators',
    multiple=True,
    help='Something found, as `name=locator`, e.g. `stake=textbox[name="Stake"]`. Repeatable.',
)
def page_fix(venue_ref: str, url: str, match: str, locators: tuple[str, ...]) -> None:
    """Pin what exploring found."""
    with _report_errors():
        session = _load_session(venue_ref)
        pinned = asyncio.run(_fix(session, url, match, _parse_locators(locators)))
        Console().print(Panel.fit(f'[bold]{pinned.match}[/bold]\n{pinned.url}\n\n{pinned.locators}'))
