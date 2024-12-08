"""Selection of the dataloader's parameters."""

from collections.abc import Callable
from typing import Self

import reflex as rx

from ...template import template
from . import DATALOADERS
from .dataloader import DataloaderState


class ParamsState(DataloaderState):
    """The parameters state."""

    param_checked: dict[str, bool] = {}
    odds_types: list[str] = []

    @staticmethod
    def process_form_data(form_data: dict[str, str]) -> list[str]:
        """Process the form data."""
        return [key.replace('"', '') for key in form_data]

    def reset_params_state(self: Self) -> None:
        """Reset the parameters state."""
        self.reset_dataloader_state()
        self.param_checked = {}
        self.odds_types = []

    def update_param_checked(self: Self, name: str | int, checked: bool) -> None:
        """Update the parameters."""
        if isinstance(name, str):
            name = f'"{name}"'
        self.param_checked[name] = checked

    def handle_submit_leagues(self: Self, leagues_form_data: list[str]) -> None:
        """Handle the form submit."""
        self.leagues = self.process_form_data(leagues_form_data)
        self.update_params()

    def handle_submit_years(self: Self, years_form_data: dict) -> None:
        """Handle the form submit."""
        self.years = [int(year) for year in self.process_form_data(years_form_data)]
        self.update_params()

    def handle_submit_divisions(self: Self, divisions_form_data: dict) -> None:
        """Handle the form submit."""
        self.divisions = [int(division) for division in self.process_form_data(divisions_form_data)]
        self.update_params()

    def update_params(self: Self) -> None:
        """Update the parameters grid."""
        self.params = [
            params
            for params in self.all_params
            if params['league'] in self.leagues
            and params['year'] in self.years
            and params['division'] in self.divisions
        ]

    def set_odds_types(self: Self) -> None:
        """Set the odds types."""
        param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
        self.odds_types = DATALOADERS[self.sport_selected](param_grid).get_odds_types()


def checkboxes(row: list[str]) -> rx.Component:
    """Checkbox of parameter value."""
    return rx.vstack(
        rx.foreach(
            row,
            lambda name: rx.checkbox(
                name,
                default_checked=True,
                checked=ParamsState.param_checked[name.to_string()],
                name=name.to_string(),
                on_change=lambda checked: ParamsState.update_param_checked(name, checked),
            ),
        ),
    )


def dialog(name: str, icon_name: str) -> Callable:
    """Dialog component."""

    def _dialog(rows: list[list[str]], on_submit: Callable) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(rx.button(rx.icon(icon_name), size='4', variant='outline')),
            rx.dialog.content(
                rx.form.root(
                    rx.dialog.title(name),
                    rx.dialog.description(
                        f'Select the {name.lower()} to include in the training data.',
                        size="2",
                        margin_bottom="16px",
                    ),
                    rx.hstack(rx.foreach(rows, checkboxes)),
                    rx.flex(
                        rx.dialog.close(rx.icon_button('check', type='submit')),
                        justify='end',
                        spacing="3",
                        margin_top="50px",
                    ),
                    on_submit=on_submit,
                    reset_on_submit=False,
                    width="100%",
                ),
            ),
        )

    return _dialog


def main_component() -> rx.Component:
    """The main component."""
    return rx.hstack(
        dialog('Leagues', 'earth')(ParamsState.all_leagues, ParamsState.handle_submit_leagues),
        dialog('Years', 'calendar')(ParamsState.all_years, ParamsState.handle_submit_years),
        dialog('Divisions', 'gauge')(ParamsState.all_divisions, ParamsState.handle_submit_divisions),
    )


def control_component() -> rx.Component:
    """The control component."""
    return rx.hstack(
        rx.cond(
            ParamsState.params,
            rx.link(rx.icon_button('check', on_click=ParamsState.set_odds_types), href='train-data-selection'),
            rx.cond(
                ParamsState.leagues,
                rx.cond(
                    ParamsState.years,
                    rx.tooltip(rx.icon_button('check', disabled=True), content='No divisions were selected.'),
                    rx.tooltip(rx.icon_button('check', disabled=True), content='No years were selected.'),
                ),
                rx.tooltip(rx.icon_button('check', disabled=True), content='No leagues were selected.'),
            ),
        ),
        rx.link(
            rx.icon_button('arrow-left', on_click=ParamsState.reset_params_state),
            href='/dataloader-selection',
        ),
    )


@rx.page(route="/dataloader-params-selection")
@template('Data', 'database', 'Parameters', 'proportions')
def dataloader_params_selection() -> rx.Component:
    """Parameters page."""
    return main_component(), control_component()
