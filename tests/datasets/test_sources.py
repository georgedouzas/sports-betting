"""Tests for the data sources."""

import pandas as pd
import pytest

from sportsbet.datasets import (
    BaseOddsSource,
    BaseSource,
    BaseStatsSource,
    PreparationReport,
    RawItem,
    RawPayload,
    SoccerDataLoader,
    from_snapshots,
)

PARAMS = [{'league': 'England', 'division': 1, 'year': 2024}, {'league': 'Spain', 'division': 1, 'year': 2024}]
ITEM_COST = 2
TOTAL_COST = 4
DISTINCT_ITEMS = 2


class _TestSource(BaseSource):
    """A concrete test source whose items and snapshots are supplied directly."""

    name = 'test'

    def index_items(self):
        return [RawItem(source=self.name, key='index', url='index.html')]

    def catalogue(self, payloads):
        return PARAMS

    def required_items(self, params):
        return [
            RawItem(
                source=self.name,
                key=f'{param["league"]}_{param["year"]}',
                url=f'{param["league"]}.csv',
                cost=ITEM_COST,
            )
            for param in params
        ]

    def to_snapshots(self, payloads):
        return pd.concat([pd.read_csv(payload.content) for payload in payloads], ignore_index=True)


class _TestStatsSource(_TestSource, BaseStatsSource):
    """A concrete test statistics source."""


class _TestOddsSource(_TestSource, BaseOddsSource):
    """A concrete test odds source."""


def test_required_items_is_deterministic():
    """Test the same parameters declare the same items in the same order."""
    source = _TestSource()
    assert source.required_items(PARAMS) == source.required_items(PARAMS)


def test_items_are_identified_by_source_and_key():
    """Test two sources declaring the same key declare the same item, so it is fetched once."""
    item = RawItem(source='football_data', key='England_1_2024', url='a.csv')
    same = RawItem(source='football_data', key='England_1_2024', url='a.csv')
    other = RawItem(source='football_data', key='Spain_1_2024', url='b.csv')
    assert len({item, same, other}) == DISTINCT_ITEMS


def test_estimate_sums_the_item_costs():
    """Test the cost of a preparation is arithmetic over the plan, not a measurement."""
    source = _TestSource()
    assert source.estimate(source.required_items(PARAMS)) == TOTAL_COST


def test_estimate_is_zero_for_free_items():
    """Test a free source charges nothing."""
    assert _TestSource().estimate([RawItem(source='test', key='a', url='a.csv')]) == 0


def test_source_cannot_be_instantiated_without_the_contract():
    """Test a source must declare its items and transform its payloads."""
    cls = BaseSource
    with pytest.raises(TypeError, match='abstract'):
        cls()


def test_stats_and_odds_sources_are_distinguishable():
    """Test the dataloader can tell a statistics source from an odds source."""
    assert isinstance(_TestStatsSource(), BaseStatsSource)
    assert not isinstance(_TestOddsSource(), BaseStatsSource)


def test_payload_keeps_the_content_verbatim():
    """Test a payload is what the source returned, not a parse of it."""
    item = RawItem(source='test', key='a', url='a.csv')
    payload = RawPayload(item=item, content=b'league,year\nEngland,2024\n')
    assert payload.content == b'league,year\nEngland,2024\n'
    assert payload.item is item


def test_sources_are_stored_unmodified():
    """Test the sources are constructor parameters, kept as given."""
    stats, odds = _TestSource(), _TestSource()
    loader = SoccerDataLoader(stats=stats, odds=odds)
    assert loader.stats is stats
    assert loader.odds is odds


def test_prepare_is_empty_without_a_source(stats, odds):
    """Test a dataloader that is not backed by a source has nothing to prepare."""
    report = from_snapshots(stats, odds).prepare()
    assert isinstance(report, PreparationReport)
    assert not report.to_fetch
