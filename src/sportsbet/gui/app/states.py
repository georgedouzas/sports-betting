"""State classes."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import cloudpickle
import nest_asyncio
import reflex as rx
from more_itertools import chunked
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid
from typing_extensions import Self

from sportsbet.datasets import SoccerDataLoader

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
VISIBILITY_LEVELS_DATALOADER_CREATION = {
    'sport': 1,
    'parameters': 2,
    'training_parameters': 3,
    'dataloader': 4,
    'control': 5,
}
VISIBILITY_LEVELS_DATALOADER_LOADING = {
    'dataloader': 2,
    'control': 3,
}

nest_asyncio.apply()


class State(rx.State):
    """The index page state."""

    # Elements
    visibility_level: int = 1
    loading: bool = False

    # Mode
    mode_category: str = 'Data'
    mode_type: str = 'Create'

    # Message
    streamed_message = ''

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message: str = """You can create or load a dataloader to grab historical
        and fixtures data. Plus, you can create or load a betting model to test how it performs
        and find value bets for upcoming games."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(0.005)
            self.streamed_message += char
            yield

    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
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

        # Message
        self.streamed_message = ''


class DataloaderCreationState(State):
    """The dataloader creation state."""

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
    data: list | None = None
    data_cols: list | None = None
    data_title: str | None = None
    loading_db: bool = False
    X_train: list | None = None
    Y_train: list | None = None
    O_train: list | None = None
    X_train_cols: list | None = None
    Y_train_cols: list | None = None
    O_train_cols: list | None = None
    X_fix: list | None = None
    Y_fix: list | None = None
    O_fix: list | None = None
    X_fix_cols: list | None = None
    Y_fix_cols: list | None = None
    O_fix_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message: str = """Begin by selecting your sport. Currently, only soccer is
        available, but more sports will be added soon!"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(0.005)
            self.streamed_message += char
            yield

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

    def switch_displayed_data_category(self: Self) -> Generator:
        """Switch the displayed data category."""
        self.loading_db = True
        yield
        if self.data in (self.X_train, self.Y_train, self.O_train):
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = 'Fixtures input data'
            self.loading_db = False
            yield
        elif self.data in (self.X_fix, self.O_fix):
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = 'Training input data'
            self.loading_db = False
            yield

    def switch_displayed_data_type(self: Self) -> Generator:
        """Switch the displayed data type."""
        self.loading_db = True
        yield
        if self.data == self.X_train:
            self.data = self.Y_train
            self.data_cols = self.Y_train_cols
            self.data_title = 'Training output data'
            self.loading_db = False
            yield
        elif self.data == self.Y_train:
            self.data = self.O_train
            self.data_cols = self.O_train_cols
            self.data_title = 'Training odds data'
            self.loading_db = False
            yield
        elif self.data == self.O_train:
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = 'Training input data'
            self.loading_db = False
            yield
        elif self.data == self.X_fix:
            self.data = self.O_fix
            self.data_cols = self.O_fix_cols
            self.data_title = 'Fixtures odds data'
            self.loading_db = False
            yield
        elif self.data == self.O_fix:
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = 'Fixtures input data'
            self.loading_db = False
            yield

    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['parameters']:
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
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(0.005)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['training_parameters']:
            self.update_params()
            self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
            self.odds_types = DATALOADERS[self.sport_selection](self.param_grid).get_odds_types()
            self.loading = False
            yield
            message = """Your training and fixtures data will include tables with odds as entries, and
            you can choose the type of odds. You can also set a tolerance level for missing values in
            the training data, which will apply to the fixtures data since it follows the same schema."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(0.005)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['dataloader']:
            dataloader = DATALOADERS[self.sport_selection](self.param_grid)
            X_train, Y_train, O_train = dataloader.extract_train_data(
                odds_type=self.odds_type,
                drop_na_thres=float(self.drop_na_thres[0]),
            )
            X_fix, _, O_fix = dataloader.extract_fixtures_data()
            self.data = self.X_train = X_train.reset_index().fillna('NA').to_dict('records')
            self.data_cols = self.X_train_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_train.columns
            ]
            self.data_title = 'Training input data'
            self.Y_train = Y_train.fillna('NA').to_dict('records')
            self.Y_train_cols = [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in Y_train.columns
            ]
            self.O_train = O_train.fillna('NA').to_dict('records') if O_train is not None else None
            self.O_train_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_train.columns]
                if O_train is not None
                else None
            )
            self.X_fix = X_fix.reset_index().fillna('NA').to_dict('records')
            self.X_fix_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_fix.columns
            ]
            self.O_fix = O_fix.fillna('NA').to_dict('records') if O_fix is not None else None
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
        self.data = None
        self.data_cols = None
        self.data_title = None
        self.loading_db = False
        self.X_train = None
        self.Y_train = None
        self.O_train = None
        self.X_train_cols = None
        self.Y_train_cols = None
        self.O_train_cols = None
        self.X_fix = None
        self.Y_fix = None
        self.O_fix = None
        self.X_fix_cols = None
        self.Y_fix_cols = None
        self.O_fix_cols = None

        # Message
        self.streamed_message = ''


class DataloaderLoadingState(State):
    """The toolbox state."""

    # Data
    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    data: list | None = None
    data_cols: list | None = None
    data_title: str | None = None
    loading_db: bool = False
    all_leagues: list[list[str]] = []  # noqa: RUF012
    all_years: list[list[str]] = []  # noqa: RUF012
    all_divisions: list[list[str]] = []  # noqa: RUF012
    param_checked: dict[str, bool] = {}  # noqa: RUF012
    odds_type: str | None = None
    drop_na_thres: float | None = None
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

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message: str = """Drag and drop or select a dataloader file to extract
        the latest training and fixtures data."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(0.005)
            self.streamed_message += char
            yield

    @rx.event
    async def handle_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(self.dataloader_serialized, 'iso8859_16')
        return rx.download(data=dataloader, filename=self.dataloader_filename)

    @staticmethod
    def process_cols(col: str) -> str:
        """Proces a column."""
        return " ".join([" ".join(token.split('_')).title() for token in col.split('__')])

    def switch_displayed_data_category(self: Self) -> Generator:
        """Switch the displayed data category."""
        self.loading_db = True
        yield
        if self.data in (self.X_train, self.Y_train, self.O_train):
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = 'Fixtures input data'
            self.loading_db = False
            yield
        elif self.data in (self.X_fix, self.O_fix):
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = 'Training input data'
            self.loading_db = False
            yield

    def switch_displayed_data_type(self: Self) -> Generator:
        """Switch the displayed data type."""
        self.loading_db = True
        yield
        if self.data == self.X_train:
            self.data = self.Y_train
            self.data_cols = self.Y_train_cols
            self.data_title = 'Training output data'
            self.loading_db = False
            yield
        elif self.data == self.Y_train:
            self.data = self.O_train
            self.data_cols = self.O_train_cols
            self.data_title = 'Training odds data'
            self.loading_db = False
            yield
        elif self.data == self.O_train:
            self.data = self.X_train
            self.data_cols = self.X_train_cols
            self.data_title = 'Training input data'
            self.loading_db = False
            yield
        elif self.data == self.X_fix:
            self.data = self.O_fix
            self.data_cols = self.O_fix_cols
            self.data_title = 'Fixtures odds data'
            self.loading_db = False
            yield
        elif self.data == self.O_fix:
            self.data = self.X_fix
            self.data_cols = self.X_fix_cols
            self.data_title = 'Fixtures input data'
            self.loading_db = False
            yield

    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_LOADING['dataloader']:
            dataloader = cloudpickle.loads(bytes(self.dataloader_serialized, 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            X_fix, _, O_fix = dataloader.extract_fixtures_data()
            self.data = self.X_train = X_train.reset_index().fillna('NA').to_dict('records')
            self.data_cols = self.X_train_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_train.columns
            ]
            self.data_title = 'Training input data'
            self.Y_train = Y_train.fillna('NA').to_dict('records')
            self.Y_train_cols = [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in Y_train.columns
            ]
            self.O_train = O_train.fillna('NA').to_dict('records') if O_train is not None else None
            self.O_train_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_train.columns]
                if O_train is not None
                else None
            )
            self.X_fix = X_fix.reset_index().fillna('NA').to_dict('records')
            self.X_fix_cols = [ag_grid.column_def(field='date', header_name='Date')] + [
                ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in X_fix.columns
            ]
            self.O_fix = O_fix.fillna('NA').to_dict('records') if O_fix is not None else None
            self.O_fix_cols = (
                [ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in O_fix.columns]
                if O_fix is not None
                else None
            )
            all_params = dataloader.get_all_params()
            self.all_leagues = list(chunked(sorted({params['league'] for params in all_params}), 6))
            self.all_years = list(chunked(sorted({params['year'] for params in all_params}), 5))
            self.all_divisions = list(chunked(sorted({params['division'] for params in all_params}), 1))
            self.param_checked = {
                **{f'"{key}"': True for key in {params['league'] for params in dataloader.param_grid_}},
                **{key: True for key in {params['year'] for params in dataloader.param_grid_}},
                **{key: True for key in {params['division'] for params in dataloader.param_grid_}},
            }
            self.odds_type = dataloader.odds_type_
            self.drop_na_thres = dataloader.drop_na_thres_
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
        self.dataloader_filename = None
        self.data = None
        self.data_cols = None
        self.data_title = None
        self.loading_db = False
        self.all_leagues = []
        self.all_years = []
        self.all_divisions = []
        self.param_checked = {}
        self.odds_type = None
        self.drop_na_thres = None
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
        self.streamed_message = ''
