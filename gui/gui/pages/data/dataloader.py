"""Selection of the dataloader."""

from itertools import batched
from typing import Any, Self

import nest_asyncio
import reflex as rx

from ...template import template
from ..index import ToolboxState
from . import DATALOADERS

nest_asyncio.apply()


class DataloaderState(ToolboxState):
    """The dataloader state."""

    sport_selected: str | None = None
    all_params: list[dict[str, Any]] = []
    all_leagues: list[list[str]] = []
    all_years: list[list[str]] = []
    all_divisions: list[list[str]] = []
    leagues: list[str] = []
    years: list[str] = []
    divisions: list[str] = []
    params: list[dict[str, Any]] = []

    def reset_dataloader_state(self: Self) -> None:
        """Reset the dataloader state."""
        self.tool = None
        self.sport_selected = None
        self.all_params = []
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []
        self.leagues = []
        self.years = []
        self.divisions = []
        self.params = []
        self.dataloader = None

    def set_sport_selected(self: Self, sport_selected: str) -> None:
        """Set the selection of sport."""
        self.sport_selected = sport_selected
        self.params = self.all_params = (
            DATALOADERS[sport_selected].get_all_params() if sport_selected is not None else []
        )
        self.all_leagues = list(batched(sorted({params['league'] for params in self.all_params}), 6))
        self.all_years = list(batched(sorted({params['year'] for params in self.all_params}), 5))
        self.all_divisions = list(batched(sorted({params['division'] for params in self.all_params}), 1))
        self.leagues = [league for row in self.all_leagues for league in row]
        self.years = [year for row in self.all_years for year in row]
        self.divisions = [division for row in self.all_divisions for division in row]


def main_component() -> rx.Component:
    """The main component."""
    return rx.select(
        ['Soccer'],
        width='50%',
        placeholder='Selection of sport',
        on_change=DataloaderState.set_sport_selected,
    )


def control_component() -> rx.Component:
    """The control component."""
    return rx.hstack(
        rx.cond(
            DataloaderState.sport_selected,
            rx.link(rx.icon_button('check'), href='dataloader-params-selection'),
            rx.tooltip(rx.icon_button('check', disabled=True), content='Please select a sport'),
        ),
        rx.link(rx.icon_button('arrow-left', on_click=DataloaderState.reset_dataloader_state), href='/'),
    )


@rx.page(route="/dataloader-selection")
@template('Data', 'database', 'Sport', 'medal')
def dataloader_selection() -> rx.Component:
    """Dataloader page."""
    return main_component(), control_component()
