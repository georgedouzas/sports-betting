"""Build a dataloader from the names of its sources and the seasons to select."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import os

from ..core import STATUSES, BuildError, ParamGrid
from ..sources import (
    BaseOddsSource,
    BaseStatsSource,
    EuroLeagueStats,
    FootballDataOdds,
    FootballDataStats,
    NBAStats,
    OddsApi,
)
from ._sourced import DataLoader

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


def _parse_moments(moments: list[str] | None) -> list[tuple[str, int]] | None:
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


def _parse_aliases(aliases: list[str] | None) -> dict[str, str] | None:
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


def _build_odds_source(
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
    if not os.environ.get(key_env):
        msg = f'`{odds}` needs a key. Set `{key_env}`, or name another variable with `--odds-key-env`.'
        raise BuildError(msg)
    return OddsApi(key_env=key_env, markets=markets or None, regions=regions or None, moments=_parse_moments(moments))


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

    Raises:
        BuildError:
            If a source name is unknown, or an argument is malformed.
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
        odds=_build_odds_source(odds, odds_key_env, odds_markets, odds_regions, odds_moments) if odds else None,
        aliases=_parse_aliases(aliases),
    )
