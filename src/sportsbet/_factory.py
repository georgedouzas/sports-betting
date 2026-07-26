"""Build a dataloader, a bettor or a venue from the strings a surface is given."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.utils import all_estimators

from . import ParamGrid
from .dataloaders import DataLoader
from .evaluation import BaseBettor, BettorGridSearchCV, ClassifierBettor, OddsComparisonBettor

if TYPE_CHECKING:
    from .execution import BaseVenue, BrowserSession

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
KEYED_SOURCES = {'odds-api'}
DEFAULT_KEY_ENV = 'ODDS_API_KEY'
STATUSES = ['preplay', 'inplay', 'postplay']
EXECUTION_EXTRA = "Placing bets needs the execution extra. Install it with `pip install 'sports-betting[execution]'`."


class BuildError(ValueError):
    """Raised when the given names do not describe something that can be built."""


def _load_object(reference: str) -> object:
    """Return the object a reference names, which is a Python file and a name inside it."""
    path, _, name = reference.rpartition(':')
    if not name:
        msg = f'`{reference}` should name an object inside a Python file, as in `models.py:BETTOR`.'
        raise BuildError(msg)
    if not Path(path).exists():
        msg = f'The file `{path}` does not exist.'
        raise BuildError(msg)
    spec = spec_from_file_location('sportsbet_model', path)
    if spec is None or spec.loader is None:
        msg = f'The file `{path}` could not be read as Python.'
        raise BuildError(msg)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, name):
        msg = f'The file `{path}` has no `{name}` in it.'
        raise BuildError(msg)
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
            raise BuildError(msg)
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
            raise BuildError(msg)
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
        raise BuildError(msg)
    if odds not in KEYED_SOURCES:
        return ODDS_SOURCES[odds]()
    key = os.environ.get(key_env)
    if not key:
        msg = f'`{odds}` needs a key. Set `{key_env}`, or name another variable with `--odds-key-env`.'
        raise BuildError(msg)
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
) -> DataLoader:
    """Build a dataloader from the names of its sources and the seasons to select.

    Args:
        stats:
            The statistics source to read, one of the ready-made names (`football-data`, `euroleague`, `nba`).
        odds:
            The odds source to pair with the statistics, or `None` for a dataloader with no odds.
        leagues:
            The leagues to select, or `None` for every league the sources publish.
        divisions:
            The divisions to select, or `None` for every division.
        years:
            The years to select, or `None` for every year.
        odds_key_env:
            The name of the environment variable holding the odds source's API key, read when the source needs one.
        odds_markets:
            The markets the odds source should price, or `None` for its own default.
        odds_regions:
            The regions the odds source should price, or `None` for its own default.
        odds_moments:
            The moments the odds source should price, each as `status:minute`, or `None` for its default.
        aliases:
            The teams the sources spell differently, each as `stats name=odds name`.

    Returns:
        dataloader:
            The dataloader that downloads and shapes the selected data.
    """
    if stats not in STATS_SOURCES:
        msg = f'`{stats}` is not a statistics source. Available: {", ".join(sorted(STATS_SOURCES))}.'
        raise BuildError(msg)
    selected: ParamGrid = {
        name: values for name, values in (('league', leagues), ('division', divisions), ('year', years)) if values
    }
    return DataLoader(
        param_grid=selected or None,
        stats=STATS_SOURCES[stats](),
        odds=_odds_source(odds, odds_key_env, odds_markets, odds_regions, odds_moments) if odds else None,
        aliases=_aliases(aliases),
    )


def build_venue(venue: str) -> BaseVenue | BrowserSession:
    """Build a venue from a reference to where it lives.

    Args:
        venue:
            Where the venue lives, as in `venue.py:VENUE`. The library ships no bookmaker: a venue with an API
            is a `BaseVenue` you write, and a bookmaker's website is a `BrowserSession` you configure.

    Returns:
        built:
            The venue, or the browser session.
    """
    try:
        from .execution import BaseVenue, BrowserSession  # noqa: PLC0415
    except ImportError as missing:
        raise BuildError(EXECUTION_EXTRA) from missing
    if ':' not in venue:
        msg = f'`{venue}` should name a venue in a Python file, as in `venue.py:VENUE`. The library ships none.'
        raise BuildError(msg)
    built = _load_object(venue)
    if not isinstance(built, BaseVenue | BrowserSession):
        msg = f'`{venue}` is not a venue and is not a browser session.'
        raise BuildError(msg)
    return built


def _bettor_namespace() -> dict[str, object]:
    """Return the estimators an inline model expression may name."""
    namespace: dict[str, object] = dict(all_estimators())
    namespace['make_pipeline'] = make_pipeline
    namespace['make_column_transformer'] = make_column_transformer
    namespace['ClassifierBettor'] = ClassifierBettor
    namespace['OddsComparisonBettor'] = OddsComparisonBettor
    namespace['BettorGridSearchCV'] = BettorGridSearchCV
    return namespace


def build_bettor(model: str) -> BaseBettor:
    """Build a betting model from a scikit-learn expression or a reference to your own.

    Args:
        model:
            A scikit-learn estimator written as a Python expression, with the library's bettors and every
            scikit-learn estimator already in scope, as in `ClassifierBettor(LogisticRegression(C=1.0))`; or a
            bettor you built in a file, named by where it lives, as in `models.py:BETTOR`.

    Returns:
        bettor:
            The betting model, ready to fit.

    Raises:
        BuildError:
            When the expression or the reference does not describe a bettor.
    """
    if ':' in model and '(' not in model:
        built = _load_object(model)
    else:
        try:
            built = eval(model, _bettor_namespace())  # noqa: S307
        except Exception as error:
            msg = (
                f'`{model}` is not a model. Write it as a scikit-learn expression, as in '
                '`OddsComparisonBettor(alpha=0.05)`, or point to one with `models.py:BETTOR`.'
            )
            raise BuildError(msg) from error
    if not isinstance(built, BaseBettor):
        msg = f'`{model}` is not a bettor.'
        raise BuildError(msg)
    return built
