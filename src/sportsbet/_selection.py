"""Implements the selection, which is what a surface is told instead of being handed a file.

A dataloader is a sport, the data to select from it, and the sources it reads. All of that is a short, closed list of
names, so it fits in the arguments of a command or of a tool, and nothing needs to be written down first.

A betting model is the exception. A model is a scikit-learn estimator, and an estimator can be any pipeline anybody can
build. The ready-made ones are named here, and anything beyond them is named by where it lives, as an object rather than
as a settings file that tries to describe one.

The command line and the server are told the same things in the same way, so neither owns a format the other has to
learn.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from . import ParamGrid
from .dataloaders import DataLoader
from .evaluation import BaseBettor, ClassifierBettor, OddsComparisonBettor

if TYPE_CHECKING:
    from .execution import BaseVenue

from .sources import (
    BaseOddsSource,
    BaseStatsSource,
    EuroLeagueStats,
    FootballDataOdds,
    FootballDataStats,
    NBAStats,
    OddsApi,
)

STATS_SOURCES: dict[str, type[BaseStatsSource]] = {
    'football-data': FootballDataStats,
    'euroleague': EuroLeagueStats,
    'nba': NBAStats,
}
ODDS_SOURCES: dict[str, type[BaseOddsSource]] = {
    'football-data': FootballDataOdds,
    'odds-api': OddsApi,
}
MODELS = ['odds-comparison', 'logistic']
VENUES = ['betfair']
KEYED_SOURCES = {'odds-api'}
DEFAULT_KEY_ENV = 'ODDS_API_KEY'
STATUSES = ['preplay', 'inplay', 'postplay']
EXECUTION_EXTRA = "Placing bets needs the execution extra. Install it with `pip install 'sports-betting[execution]'`."


class SelectionError(ValueError):
    """Raised when what a surface was told does not describe something that can be built."""


def _load_object(reference: str) -> object:
    """Return the object a reference names, which is a Python file and a name inside it."""
    path, _, name = reference.rpartition(':')
    if not name:
        msg = f'`{reference}` should name an object inside a Python file, as in `models.py:BETTOR`.'
        raise SelectionError(msg)
    if not Path(path).exists():
        msg = f'The file `{path}` does not exist.'
        raise SelectionError(msg)
    spec = spec_from_file_location('sportsbet_model', path)
    if spec is None or spec.loader is None:
        msg = f'The file `{path}` could not be read as Python.'
        raise SelectionError(msg)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, name):
        msg = f'The file `{path}` has no `{name}` in it.'
        raise SelectionError(msg)
    return getattr(mod, name)


def _moments(moments: list[str] | None) -> list[tuple[str, int]] | None:
    """Return the moments to price, each of them a status and how many minutes into the match it is."""
    if not moments:
        return None
    parsed = []
    for moment in moments:
        status, _, minutes = moment.partition(':')
        if status not in STATUSES or not minutes.isdigit():
            msg = f'`{moment}` should be a status and a minute, as in `inplay:45`.'
            raise SelectionError(msg)
        parsed.append((status, int(minutes)))
    return parsed


def _aliases(aliases: list[str] | None) -> dict[str, str] | None:
    """Return the teams the two sources spell differently, each of them a name and the other name."""
    if not aliases:
        return None
    paired = {}
    for alias in aliases:
        stats_name, sep, odds_name = alias.partition('=')
        if not sep or not stats_name or not odds_name:
            msg = f'`{alias}` should be two names, as in `Olimpia Milano=EA7 Emporio Armani Milan`.'
            raise SelectionError(msg)
        paired[stats_name] = odds_name
    return paired


def _odds_source(
    odds: str,
    key_env: str,
    markets: list[str] | None,
    regions: list[str] | None,
    moments: list[str] | None,
) -> BaseOddsSource:
    """Return the odds source a name asks for, reading a key from the environment when it needs one."""
    if odds not in ODDS_SOURCES:
        msg = f'`{odds}` is not an odds source. Available: {", ".join(sorted(ODDS_SOURCES))}.'
        raise SelectionError(msg)
    if odds not in KEYED_SOURCES:
        return ODDS_SOURCES[odds]()
    key = os.environ.get(key_env)
    if not key:
        msg = f'`{odds}` needs a key. Set `{key_env}`, or name another variable with `--odds-key-env`.'
        raise SelectionError(msg)
    return OddsApi(key=key, markets=markets or None, regions=regions or None, moments=_moments(moments))


def build_dataloader(
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
) -> DataLoader:
    """Return the dataloader a selection describes.

    The statistics have to be named. Which feed the data came from decides what is in it, what it costs and whether
    anyone may redistribute it, so you name it yourself. It also decides the sport, so the sport comes from the source
    rather than a separate argument.

    The odds are optional. With no odds you get the features on their own, which is enough to explore the data or to
    learn from it without a target of ours.
    """
    if stats not in STATS_SOURCES:
        msg = f'`{stats}` is not a statistics source. Available: {", ".join(sorted(STATS_SOURCES))}.'
        raise SelectionError(msg)
    selected: ParamGrid = {
        name: values for name, values in (('league', leagues), ('division', divisions), ('year', years)) if values
    }
    return DataLoader(
        param_grid=selected or None,
        stats=STATS_SOURCES[stats](),
        odds=_odds_source(odds, odds_key_env, odds_markets, odds_regions, odds_moments) if odds else None,
        aliases=_aliases(aliases),
        max_unmatched_rate=max_unmatched_rate,
    )


def build_venue(venue: str) -> BaseVenue:
    """Return the venue a selection describes.

    A venue is named the way a model is named rather than the way a dataloader is. A dataloader is a sport and a short
    list of names, so it fits in arguments. A venue carries URLs, notes and certificates, and a site-driven one carries
    prose, so it is named by where it lives and built in Python.

    The import is made here rather than at the top of the file because placing lives behind an optional extra, and this
    module is imported on every run.

    Args:
        venue:
            A ready-made venue, or where one of your own lives, as in `venue.py:VENUE`.

    Returns:
        built:
            The venue.
    """
    try:
        from .execution import BaseVenue, BetfairVenue  # noqa: PLC0415
    except ImportError as missing:
        raise SelectionError(EXECUTION_EXTRA) from missing
    if ':' in venue:
        built = _load_object(venue)
        if not isinstance(built, BaseVenue):
            msg = f'`{venue}` is not a venue.'
            raise SelectionError(msg)
        return built
    if venue == 'betfair':
        return BetfairVenue()
    msg = f'`{venue}` is not a venue. Ready-made: {", ".join(VENUES)}. For one of your own, use `venue.py:VENUE`.'
    raise SelectionError(msg)


def build_bettor(
    model: str,
    alpha: float = 0.05,
    init_cash: float | None = None,
    stake: float | None = None,
    betting_markets: list[str] | None = None,
    model_odds_types: list[str] | None = None,
) -> BaseBettor:
    """Return the betting model a selection describes.

    A ready-made model is named. Anything else is a scikit-learn estimator, so it is named by where it lives —
    `models.py:BETTOR` — and it is built in Python, where it belongs.
    """
    markets = betting_markets or None
    if ':' in model:
        built = _load_object(model)
        if not isinstance(built, BaseBettor):
            msg = f'`{model}` is not a bettor.'
            raise SelectionError(msg)
        return built
    if model == 'odds-comparison':
        return OddsComparisonBettor(
            odds_types=model_odds_types or None,
            alpha=alpha,
            betting_markets=markets,
            init_cash=init_cash,
            stake=stake,
        )
    if model == 'logistic':
        classifier = make_pipeline(
            make_column_transformer(
                (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                remainder='passthrough',
            ),
            SimpleImputer(),
            MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
        )
        return ClassifierBettor(classifier, betting_markets=markets, init_cash=init_cash, stake=stake)
    msg = f'`{model}` is not a model. Ready-made: {", ".join(MODELS)}. For one of your own, use `models.py:BETTOR`.'
    raise SelectionError(msg)
