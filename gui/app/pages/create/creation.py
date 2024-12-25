"""Index page."""

from itertools import batched
from typing import Any, Self

import cloudpickle
import nest_asyncio
import reflex as rx
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid

from sportsbet.datasets import SoccerDataLoader

from ...components.dataloader.creation import main

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
DEFAULT_STATE_VALS = {
    'mode': {
        'category': 'Data',
        'type': 'Create',
    },
    'sport': {
        'selection': 'Soccer',
        'all_params': [],
        'all_leagues': [],
        'all_years': [],
        'all_divisions': [],
        'leagues': [],
        'years': [],
        'divisions': [],
        'params': [],
    },
    'parameters': {
        'checked': {},
        'default_checked': DEFAULT_PARAM_CHECKED,
        'odds_types': [],
        'param_grid': [],
    },
    'training_parameters': {
        'odds_type': 'market_average',
        'drop_na_thres': [0.0],
    },
    'data': {
        'X_train': None,
        'Y_train': None,
        'O_train': None,
        'X_train_cols': None,
        'Y_train_cols': None,
        'O_train_cols': None,
        'X_fix': None,
        'O_fix': None,
        'X_fix_cols': None,
        'O_fix_cols': None,
    },
}

nest_asyncio.apply()


class DataloaderCreationState(rx.State):
    """The toolbox state."""

    # Elements
    visibility_level: int = 1
    loading: bool = False

    # Mode
    mode_category: str = DEFAULT_STATE_VALS['mode']['category']
    mode_type: str = DEFAULT_STATE_VALS['mode']['type']

    # Sport
    sport_selection: str = DEFAULT_STATE_VALS['sport']['selection']
    all_params: list[dict[str, Any]] = DEFAULT_STATE_VALS['sport']['all_params']
    all_leagues: list[list[str]] = DEFAULT_STATE_VALS['sport']['all_leagues']
    all_years: list[list[str]] = DEFAULT_STATE_VALS['sport']['all_years']
    all_divisions: list[list[str]] = DEFAULT_STATE_VALS['sport']['all_divisions']
    leagues: list[str] = DEFAULT_STATE_VALS['sport']['leagues']
    years: list[str] = DEFAULT_STATE_VALS['sport']['years']
    divisions: list[str] = DEFAULT_STATE_VALS['sport']['divisions']
    params: list[dict[str, Any]] = DEFAULT_STATE_VALS['sport']['params']

    # Parameters
    param_checked: dict[str, bool] = DEFAULT_STATE_VALS['parameters']['checked']
    default_param_checked: dict[str, list[str]] = DEFAULT_STATE_VALS['parameters']['default_checked']
    odds_types: list[str] = DEFAULT_STATE_VALS['parameters']['odds_types']
    param_grid: list[dict] = DEFAULT_STATE_VALS['parameters']['param_grid']

    # Training parameters
    odds_type: str = DEFAULT_STATE_VALS['training_parameters']['odds_type']
    drop_na_thres: list = DEFAULT_STATE_VALS['training_parameters']['drop_na_thres']

    # Data
    dataloader_serialized: str | None = None
    X_train: list | None = DEFAULT_STATE_VALS['data']['X_train']
    Y_train: list | None = DEFAULT_STATE_VALS['data']['Y_train']
    O_train: list | None = DEFAULT_STATE_VALS['data']['O_train']
    X_train_cols: list | None = DEFAULT_STATE_VALS['data']['X_train_cols']
    Y_train_cols: list | None = DEFAULT_STATE_VALS['data']['Y_train_cols']
    O_train_cols: list | None = DEFAULT_STATE_VALS['data']['O_train_cols']
    X_fix: list | None = DEFAULT_STATE_VALS['data']['X_fix']
    O_fix: list | None = DEFAULT_STATE_VALS['data']['O_fix']
    X_fix_cols: list | None = DEFAULT_STATE_VALS['data']['X_fix_cols']
    O_fix_cols: list | None = DEFAULT_STATE_VALS['data']['O_fix_cols']

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

    async def submit_state(self: Self) -> None:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        elif self.visibility_level == 2:
            self.all_params = DATALOADERS[self.sport_selection].get_all_params()
            self.all_leagues = list(batched(sorted({params['league'] for params in self.all_params}), 6))
            self.all_years = list(batched(sorted({params['year'] for params in self.all_params}), 5))
            self.all_divisions = list(batched(sorted({params['division'] for params in self.all_params}), 1))
            self.leagues = [league.replace('"', '') for league in DEFAULT_PARAM_CHECKED['leagues']]
            self.years = [int(year) for year in DEFAULT_PARAM_CHECKED['years']]
            self.divisions = [int(division) for division in DEFAULT_PARAM_CHECKED['divisions']]
            self.loading = False
            yield
        elif self.visibility_level == 3:
            self.update_params()
            self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
            self.odds_types = DATALOADERS[self.sport_selection](self.param_grid).get_odds_types()
            self.loading = False
            yield
        elif self.visibility_level == 4:
            dataloader = DATALOADERS[self.sport_selection](self.param_grid)
            X_train, Y_train, O_train = dataloader.extract_train_data(
                odds_type=self.odds_type,
                drop_na_thres=self.drop_na_thres[0],
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
        self.mode_category = DEFAULT_STATE_VALS['mode']['category']
        self.mode_type = DEFAULT_STATE_VALS['mode']['type']

        # Sport
        self.sport_selection = DEFAULT_STATE_VALS['sport']['selection']
        self.all_params = DEFAULT_STATE_VALS['sport']['all_params']
        self.all_leagues = DEFAULT_STATE_VALS['sport']['all_leagues']
        self.all_years = DEFAULT_STATE_VALS['sport']['all_years']
        self.all_divisions = DEFAULT_STATE_VALS['sport']['all_divisions']
        self.leagues = DEFAULT_STATE_VALS['sport']['leagues']
        self.years = DEFAULT_STATE_VALS['sport']['years']
        self.divisions = DEFAULT_STATE_VALS['sport']['divisions']
        self.params = DEFAULT_STATE_VALS['sport']['params']

        # Parameters
        self.param_checked = DEFAULT_STATE_VALS['parameters']['checked']
        self.default_param_checked = DEFAULT_STATE_VALS['parameters']['default_checked']
        self.odds_types = DEFAULT_STATE_VALS['parameters']['odds_types']
        self.param_grid = DEFAULT_STATE_VALS['parameters']['param_grid']

        # Training
        self.odds_type = DEFAULT_STATE_VALS['training_parameters']['odds_type']
        self.drop_na_thres = DEFAULT_STATE_VALS['training_parameters']['drop_na_thres']

        # Data
        self.dataloader_serialized = None
        self.X_train = DEFAULT_STATE_VALS['data']['X_train']
        self.Y_train = DEFAULT_STATE_VALS['data']['Y_train']
        self.O_train = DEFAULT_STATE_VALS['data']['O_train']
        self.X_train_cols = DEFAULT_STATE_VALS['data']['X_train_cols']
        self.Y_train_cols = DEFAULT_STATE_VALS['data']['Y_train_cols']
        self.O_train_cols = DEFAULT_STATE_VALS['data']['O_train_cols']
        self.X_fix = DEFAULT_STATE_VALS['data']['X_fix']
        self.O_fix = DEFAULT_STATE_VALS['data']['O_fix']
        self.X_fix_cols = DEFAULT_STATE_VALS['data']['X_fix_cols']
        self.O_fix_cols = DEFAULT_STATE_VALS['data']['O_fix_cols']


@rx.page(route="/dataloader/creation")
def dataloader_creation() -> rx.Component:
    """Main page."""
    return main(DataloaderCreationState)
