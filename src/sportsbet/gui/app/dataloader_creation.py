"""Dataloader creation page."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

import cloudpickle
import nest_asyncio
import reflex as rx
from more_itertools import chunked
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid
from typing_extensions import Self

from sportsbet.datasets import SoccerDataLoader

from .components import SIDEBAR_OPTIONS, control_buttons, home, select_mode, title
from .index import State

DATALOADERS = {
    'Soccer': SoccerDataLoader,
}
DEFAULT_PARAM_CHECKED = {
    'leagues': [
        '"England"',
        '"Scotland"',
        '"Germany"',
        '"Italy"',
        '"Spain"',
        '"France"',
        '"Netherlands"',
        '"Belgium"',
        '"Portugal"',
        '"Turkey"',
        '"Greece"',
    ],
    'years': [
        '2020',
        '2021',
        '2022',
        '2023',
        '2024',
        '2025',
    ],
    'divisions': ['1', '2'],
}
VISIBILITY_LEVELS = {
    'sport': 1,
    'parameters': 2,
    'training_parameters': 3,
    'dataloader': 4,
    'control': 5,
}

nest_asyncio.apply()


class DataloaderCreationState(State):
    """The toolbox state."""

    # Sport
    sport_selection: str = 'Soccer'
    all_params: list[dict[str, Any]] = []  # noqa: RUF012
    all_leagues: list[list[str]] = []  # noqa: RUF012
    all_years: list[list[str]] = []  # noqa: RUF012
    all_divisions: list[list[str]] = []  # noqa: RUF012
    leagues: list[str] = []  # noqa: RUF012
    years: list[int] = []  # noqa: RUF012
    divisions: list[int] = []  # noqa: RUF012
    params: list[dict[str, Any]] = []  # noqa: RUF012

    # Parameters
    param_checked: dict[str | int, bool] = {}  # noqa: RUF012
    default_param_checked: dict[str, list[str]] = DEFAULT_PARAM_CHECKED
    odds_types: list[str] = []  # noqa: RUF012
    param_grid: list[dict] = []  # noqa: RUF012

    # Training parameters
    odds_type: str = 'market_average'
    drop_na_thres: list = [0.0]  # noqa: RUF012

    # Data
    dataloader_serialized: str | None = None
    X_train: list | None = None
    Y_train: list | None = None
    O_train: list | None = None
    X_train_cols: list | None = None
    Y_train_cols: list | None = None
    O_train_cols: list | None = None
    X_fix: list | None = None
    O_fix: list | None = None
    X_fix_cols: list | None = None
    O_fix_cols: list | None = None

    def set_mode_category(self: Self, mode_category: str) -> None:
        """Set the mode category."""
        self.mode_category = mode_category

    def set_mode_type(self: Self, mode_type: str) -> None:
        """Set the mode category."""
        self.mode_type = mode_type

    def set_sport_selection(self: Self, sport_selection: str) -> None:
        """Set the sport."""
        self.sport_selection = sport_selection

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(self.dataloader_serialized, 'iso8859_16')
        return rx.download(data=dataloader, filename='dataloader.pkl')

    @staticmethod
    def process_cols(col: str) -> str:
        """Proces a column."""
        return " ".join([" ".join(token.split('_')).title() for token in col.split('__')])

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

    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS['parameters']:
            self.all_params = DATALOADERS[self.sport_selection].get_all_params()
            self.all_leagues = list(chunked(sorted({params['league'] for params in self.all_params}), 6))
            self.all_years = list(chunked(sorted({params['year'] for params in self.all_params}), 5))
            self.all_divisions = list(chunked(sorted({params['division'] for params in self.all_params}), 1))
            self.leagues = [league.replace('"', '') for league in DEFAULT_PARAM_CHECKED['leagues']]
            self.years = [int(year) for year in DEFAULT_PARAM_CHECKED['years']]
            self.divisions = [int(division) for division in DEFAULT_PARAM_CHECKED['divisions']]
            self.loading = False
            yield
            message = """You can choose the leagues, divisions, and years to include in the training data.
            This selection won't impact the fixtures data."""
            self.streamed_message_dataloader_creation = ''
            for char in message:
                await asyncio.sleep(0.005)
                self.streamed_message_dataloader_creation += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS['training_parameters']:
            self.update_params()
            self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
            self.odds_types = DATALOADERS[self.sport_selection](self.param_grid).get_odds_types()
            self.loading = False
            yield
            message = """Your training and fixtures data will include tables with odds as entries, and
            you can choose the type of odds. You can also set a tolerance level for missing values in
            the training data, which will apply to the fixtures data since it follows the same schema."""
            self.streamed_message_dataloader_creation = ''
            for char in message:
                await asyncio.sleep(0.005)
                self.streamed_message_dataloader_creation += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS['dataloader']:
            dataloader = DATALOADERS[self.sport_selection](self.param_grid)
            X_train, Y_train, O_train = dataloader.extract_train_data(
                odds_type=self.odds_type,
                drop_na_thres=float(self.drop_na_thres[0]),
            )
            X_fix, _, O_fix = dataloader.extract_fixtures_data()
            self.X_train = X_train.reset_index().to_dict('records')
            self.X_train_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_train.columns
            ]
            self.Y_train = Y_train.to_dict('records')
            self.Y_train_cols = [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in Y_train.columns
            ]
            self.O_train = O_train.to_dict('records') if O_train is not None else None
            self.O_train_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_train.columns]
                if O_train is not None
                else None
            )
            self.X_fix = X_fix.reset_index().to_dict('records')
            self.X_fix_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_fix.columns
            ]
            self.O_fix = O_fix.to_dict('records') if O_fix is not None else None
            self.O_fix_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_fix.columns]
                if O_fix is not None
                else None
            )
            self.dataloader_serialized = str(cloudpickle.dumps(dataloader), 'iso8859_16')
            self.loading = False
            yield
        self.visibility_level += 1

    def reset_state(self: Self) -> None:
        """Reset handler."""

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'

        # Data
        self.dataloader_serialized = None

        # Sport
        self.sport_selection = 'Soccer'
        self.all_params = []
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []
        self.leagues = []
        self.years = []
        self.divisions = []
        self.params = []

        # Parameters
        self.param_checked = {}
        self.default_param_checked = DEFAULT_PARAM_CHECKED
        self.odds_types = []
        self.param_grid = []

        # Training
        self.odds_type = 'market_average'
        self.drop_na_thres = [0.0]

        # Data
        self.X_train = None
        self.Y_train = None
        self.O_train = None
        self.X_train_cols = None
        self.Y_train_cols = None
        self.O_train_cols = None
        self.X_fix = None
        self.O_fix = None
        self.X_fix_cols = None
        self.O_fix_cols = None

        # Message
        self.streamed_message = """You can create or load a dataloader to grab historical
        and fixtures data. Plus, you can create or load a betting model to test how it performs
        and find value bets for upcoming games."""
        self.streamed_message_dataloader_creation = """Begin by selecting your sport. Currently, only soccer
        is available, but more sports will be added soon!"""
        self.streamed_message_dataloader_loading = """Drag and drop or select a dataloader file to extract
        the latest training and fixtures data."""


def checkboxes(row: list[str], state: rx.State) -> rx.Component:
    """Checkbox of parameter value."""

    def _in_leagues(name: rx.Var) -> rx.Var:
        return state.default_param_checked['leagues'].contains(name.to_string())

    def _in_years(name: rx.Var) -> rx.Var:
        return state.default_param_checked['years'].contains(name.to_string())

    def _in_divisions(name: rx.Var) -> rx.Var:
        return state.default_param_checked['divisions'].contains(name.to_string())

    return rx.vstack(
        rx.foreach(
            row,
            lambda name: rx.checkbox(
                name,
                default_checked=rx.cond(
                    _in_leagues(name),
                    True,
                    rx.cond(_in_years(name), True, rx.cond(_in_divisions(name), True, False)),
                ),
                checked=state.param_checked[name.to_string()],
                name=name.to_string(),
                on_change=lambda checked: state.update_param_checked(name, checked),
            ),
        ),
    )


def dialog(name: str, icon_name: str, state: rx.State) -> Callable:
    """Dialog component."""

    def _dialog(rows: list[list[str]], on_submit: Callable) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    rx.tooltip(rx.icon(icon_name), content=name),
                    size='4',
                    variant='outline',
                    disabled=state.visibility_level > VISIBILITY_LEVELS['training_parameters'],
                ),
            ),
            rx.dialog.content(
                rx.form.root(
                    rx.dialog.title(name),
                    rx.dialog.description(
                        f'Select the {name.lower()} to include in the training data.',
                        size="2",
                        margin_bottom="16px",
                    ),
                    rx.hstack(rx.foreach(rows, lambda row: checkboxes(row, state))),
                    rx.flex(
                        rx.dialog.close(rx.button('Submit', type='submit')),
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


def training_parameters_selection(state: rx.State) -> rx.Component:
    """The training parameters selection component."""
    return rx.vstack(
        rx.vstack(
            rx.text('Odds type', size='1'),
            rx.select(
                state.odds_types,
                default_value=state.odds_types[0],
                on_change=state.handle_odds_type,
                disabled=state.visibility_level > VISIBILITY_LEVELS['dataloader'],
                width='100%',
            ),
            style={
                'margin-top': '5px',
            },
        ),
        rx.vstack(
            rx.text(f'Drop NA threshold of columns: {DataloaderCreationState.drop_na_thres}', size='1'),
            rx.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                default_value=0.0,
                on_change=state.handle_drop_na_thres,
                disabled=state.visibility_level > VISIBILITY_LEVELS['dataloader'],
                width='200px',
            ),
            style={
                'margin-top': '5px',
            },
        ),
    )


def parameters_selection(state: rx.State) -> rx.Component:
    """The parameters title."""
    return rx.hstack(
        dialog('Leagues', 'earth', state)(state.all_leagues, state.handle_submit_leagues),
        dialog('Years', 'calendar', state)(state.all_years, state.handle_submit_years),
        dialog('Divisions', 'gauge', state)(state.all_divisions, state.handle_submit_divisions),
    )


@rx.page(route="/dataloader/creation")
def dataloader_creation_page() -> rx.Component:
    """Main page."""
    return rx.container(
        rx.vstack(
            home(),
            rx.divider(),
            # Mode selection
            title('Mode', 'blend'),
            select_mode(DataloaderCreationState, 'Create a dataloader'),
            # Sport selection
            rx.cond(
                DataloaderCreationState.visibility_level > 1,
                title('Sport', 'medal'),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level > 1,
                rx.text('Select a sport', size='1'),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level > 1,
                rx.select(
                    items=['Soccer'],
                    value='Soccer',
                    disabled=DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['parameters'],
                    on_change=DataloaderCreationState.set_sport_selection,
                    width='120px',
                ),
            ),
            # Parameters selection
            rx.cond(
                DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['parameters'],
                title('Parameters', 'proportions'),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['parameters'],
                rx.text('Select parameters', size='1'),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['parameters'],
                parameters_selection(DataloaderCreationState),
            ),
            # Training parameters selection
            rx.cond(
                DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['training_parameters'],
                training_parameters_selection(DataloaderCreationState),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level > VISIBILITY_LEVELS['dataloader'],
                rx.button(
                    'Save',
                    position='fixed',
                    top='620px',
                    left='275px',
                    width='70px',
                    on_click=DataloaderCreationState.download_dataloader,
                ),
            ),
            # Control
            control_buttons(
                DataloaderCreationState,
                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
            ),
            **SIDEBAR_OPTIONS,
        ),
        rx.vstack(
            rx.cond(
                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                rx.hstack(
                    rx.heading(
                        'Training data',
                        size='7',
                        position='fixed',
                        left='450px',
                        top='50px',
                        color_scheme='blue',
                    ),
                ),
            ),
            rx.hstack(
                rx.vstack(
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        rx.heading('Input'),
                    ),
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        ag_grid(
                            id='X_train',
                            row_data=DataloaderCreationState.X_train,
                            column_defs=DataloaderCreationState.X_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        rx.heading('Output'),
                    ),
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        ag_grid(
                            id='Y_train',
                            row_data=DataloaderCreationState.Y_train,
                            column_defs=DataloaderCreationState.Y_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        rx.heading('Odds'),
                    ),
                    rx.cond(
                        DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                        ag_grid(
                            id='O_train',
                            row_data=DataloaderCreationState.O_train,
                            column_defs=DataloaderCreationState.O_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                position='fixed',
                left='450px',
                top='100px',
            ),
        ),
        rx.vstack(
            rx.cond(
                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                rx.hstack(
                    rx.heading(
                        'Fixtures data',
                        size='7',
                        position='fixed',
                        left='450px',
                        top='370px',
                        color_scheme='blue',
                    ),
                ),
            ),
            rx.cond(
                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                rx.cond(
                    DataloaderCreationState.X_fix,
                    rx.hstack(
                        rx.vstack(
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                rx.heading('Input'),
                            ),
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                ag_grid(
                                    id='X_fix',
                                    row_data=DataloaderCreationState.X_fix,
                                    column_defs=DataloaderCreationState.X_fix_cols,
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        rx.vstack(
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                rx.heading('Output'),
                            ),
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                ag_grid(
                                    id='Y_fix',
                                    row_data=[],
                                    column_defs=[],
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        rx.vstack(
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                rx.heading('Odds'),
                            ),
                            rx.cond(
                                DataloaderCreationState.visibility_level == VISIBILITY_LEVELS['control'],
                                ag_grid(
                                    id='O_fix',
                                    row_data=DataloaderCreationState.O_fix,
                                    column_defs=DataloaderCreationState.O_fix_cols,
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        position='fixed',
                        left='450px',
                        top='420px',
                    ),
                    rx.tooltip(
                        rx.icon('ban', position='fixed', left='450px', top='420px', size=60),
                        content='No fixtures were found. Try again later.',
                    ),
                ),
            ),
        ),
        rx.cond(
            DataloaderCreationState.visibility_level < VISIBILITY_LEVELS['control'],
            rx.box(
                rx.vstack(
                    rx.icon('bot-message-square', size=70),
                    rx.html(DataloaderCreationState.streamed_message_dataloader_creation),
                ),
                position='fixed',
                left='600px',
                top='100px',
                width='500px',
                background_color=rx.color('gray', 3),
                padding='30px',
            ),
        ),
    )
