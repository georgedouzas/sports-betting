"""Index page."""

from itertools import batched
from typing import Any, Self

import nest_asyncio
import reflex as rx

from sportsbet.datasets import SoccerDataLoader

from ..components.common import header, home, selection, title
from ..components.parameters import parameters_selection
from ..components.training_parameters import training_parameters_selection

SIDEBAR_OPTIONS = {
    'spacing': '5',
    'position': 'fixed',
    'left': '50px',
    'top': '200px',
    'padding_x': '1em',
    'padding_y': "1.5em",
    'bg': rx.color('blue', 3),
    'height': '750px',
    'width': '35em',
}
DATALOADERS = {
    'Soccer': SoccerDataLoader,
}
DEFAULT_PARAM_CHECKED = {
    '"England"': True,
    '"Scotland"': True,
    '"Germany"': True,
    '"Italy"': True,
    '"Spain"': True,
    '"France"': True,
    '"Netherlands"': True,
    '"Belgium"': True,
    '"Portugal"': True,
    '"Turkey"': True,
    '"Greece"': True,
    2018: True,
    2019: True,
    2020: True,
    2021: True,
    2022: True,
    2023: True,
    2024: True,
    2025: True,
    1: True,
    2: True,
}

nest_asyncio.apply()


class State(rx.State):
    """The toolbox state."""

    # Task
    task: str | None = None
    task_disabled: bool = False

    # Sport
    sport: str | None = None
    sport_disabled: bool = False
    all_params: list[dict[str, Any]] = []
    all_leagues: list[list[str]] = []
    all_years: list[list[str]] = []
    all_divisions: list[list[str]] = []
    leagues: list[str] = []
    years: list[str] = []
    divisions: list[str] = []
    params: list[dict[str, Any]] = []

    # Parameters
    parameters_disabled: bool = False
    parameters_loading: bool = False
    param_checked: dict[str, bool] = {}
    default_param_checked: dict[str, bool] = DEFAULT_PARAM_CHECKED
    odds_types: list[str] = []
    param_grid: list[dict] = []

    # Training
    training_disabled = False
    training_loading: bool = False
    odds_type: str | None = None
    drop_na_thres: list | None = [0.0]

    @staticmethod
    def process_form_data(form_data: dict[str, str]) -> list[str]:
        """Process the form data."""
        return [key.replace('"', '') for key in form_data]

    def update_param_checked(self: Self, name: str | int, checked: bool) -> None:
        """Update the parameters."""
        if isinstance(name, str):
            name = f'"{name}"'
        self.param_checked[name] = checked

    def update_params(self: Self) -> None:
        """Update the parameters grid."""
        self.params = [
            params
            for params in self.all_params
            if params['league'] in self.leagues
            and params['year'] in self.years
            and params['division'] in self.divisions
        ]

    def handle_submit_leagues(self: Self, leagues_form_data: dict) -> None:
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

    def handle_odds_type(self, odds_type: str) -> None:
        """Handle the odds type selection."""
        self.odds_type = odds_type

    def handle_drop_na_thres(self, drop_na_thres: list) -> None:
        """Handle the drop NA threshold selection."""
        self.drop_na_thres = drop_na_thres

    def reset_state(self: Self) -> None:
        """Reset the dataloader state."""

        # Task
        self.task = None
        self.task_disabled = False

        # Sport
        self.sport = None
        self.sport_disabled = False
        self.all_params = []
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []

        # Parameters
        self.parameters_disabled = False
        self.parameters_loading = False
        self.leagues = []
        self.years = []
        self.divisions = []
        self.params = []
        self.param_checked = {}
        self.odds_types = []
        self.param_grid = []

        # Training
        self.training_disabled = False
        self.training_loading = False
        self.odds_type = None
        self.drop_na_thres = None

    def set_task(self: Self, task: str) -> None:
        """Set the task."""
        self.task = task
        self.task_disabled = True

    def set_sport(self: Self, sport: str) -> None:
        """Set the sport."""
        self.sport = sport
        self.sport_disabled = True
        yield
        self.all_params = DATALOADERS[self.sport].get_all_params()
        self.all_leagues = list(batched(sorted({params['league'] for params in self.all_params}), 6))
        self.all_years = list(batched(sorted({params['year'] for params in self.all_params}), 5))
        self.all_divisions = list(batched(sorted({params['division'] for params in self.all_params}), 1))
        self.leagues = [league for row in self.all_leagues for league in row]
        self.years = [year for row in self.all_years for year in row]
        self.divisions = [division for row in self.all_divisions for division in row]

    def set_parameters(self: Self) -> None:
        """Set the parameters."""
        self.parameters_disabled = True
        self.parameters_loading = True
        yield
        self.update_params()
        self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
        self.odds_types = DATALOADERS[self.sport](self.param_grid).get_odds_types()

    def set_training_parameters(self: Self) -> None:
        """Set the training parameters."""
        self.training_disabled = True
        self.training_loading = True
        yield
        X_train, Y_train, O_train = DATALOADERS[self.sport](self.param_grid).extract_train_data(
            odds_type=self.odds_type, drop_na_thres=self.drop_na_thres[0]
        )
        self.training_loading = False


@rx.page(route="/")
def index() -> rx.Component:
    """Main page."""
    return rx.box(
        header(),
        rx.box(
            rx.vstack(
                home(State.reset_state),
                rx.divider(),
                # Task selection
                title('Task', 'arrow-up-down'),
                rx.text('Data or modelling task selection', size='1', hidden=State.task_disabled),
                selection(['Data', 'Modelling'], State.task, State.task_disabled, State.set_task),
                # Sport selection
                rx.cond(
                    State.task == 'Data',
                    title('Sport', 'medal'),
                ),
                rx.cond(
                    State.task == 'Data',
                    rx.text('Sport selection', size='1', hidden=State.sport_disabled),
                ),
                rx.cond(
                    State.task == 'Data',
                    selection(['Soccer'], State.sport, State.sport_disabled, State.set_sport),
                ),
                # Parameters selection
                rx.cond(
                    State.sport == 'Soccer',
                    title('Parameters', 'proportions'),
                ),
                rx.cond(
                    State.sport == 'Soccer',
                    parameters_selection(State),
                ),
                rx.cond(
                    State.sport == 'Soccer',
                    rx.cond(
                        State.odds_types.to_string() == '[]',
                        rx.button(
                            'Submit',
                            on_click=State.set_parameters,
                            loading=State.parameters_loading,
                            disabled=State.parameters_disabled,
                        ),
                    ),
                ),
                # Training parameters selection
                rx.cond(
                    State.odds_types,
                    training_parameters_selection(State),
                ),
                rx.cond(
                    State.odds_types,
                    rx.button(
                        'Submit',
                        on_click=State.set_training_parameters,
                        loading=State.training_loading,
                        disabled=State.training_disabled,
                    ),
                ),
                # Options
                **SIDEBAR_OPTIONS,
            ),
        ),
    )
