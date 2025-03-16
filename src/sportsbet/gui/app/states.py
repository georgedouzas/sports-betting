"""State classes."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any, cast

import cloudpickle
import nest_asyncio
import numpy as np
import pandas as pd
import reflex as rx
from more_itertools import chunked
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Self

from sportsbet.datasets import BaseDataLoader, SoccerDataLoader
from sportsbet.evaluation import BaseBettor, BettorGridSearchCV, ClassifierBettor, OddsComparisonBettor, backtest

BETTING_MARKETS = [
    [
        'home_win__full_time_goals',
        'away_win__full_time_goals',
        'draw__full_time_goals',
        'over_2.5__full_time_goals',
        'under_2.5__full_time_goals',
    ],
    ['draw__full_time_goals', 'over_2.5__full_time_goals'],
    ['home_win__full_time_goals', 'away_win__full_time_goals'],
]
DATALOADERS = {
    'Soccer': SoccerDataLoader,
}
MODELS = {
    'Odds Comparison': BettorGridSearchCV(
        estimator=OddsComparisonBettor(),
        param_grid={
            'alpha': np.linspace(0.0, 0.05, 20),
            'betting_markets': BETTING_MARKETS,
        },
        error_score='raise',
    ),
    'Logistic Regression': BettorGridSearchCV(
        estimator=ClassifierBettor(
            classifier=make_pipeline(
                make_column_transformer(
                    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                    remainder='passthrough',
                    force_int_remainder_cols=False,
                ),
                SimpleImputer(),
                MultiOutputClassifier(
                    LogisticRegression(solver='liblinear', random_state=7, max_iter=int(1e5)),
                ),
            ),
        ),
        param_grid={
            'betting_markets': BETTING_MARKETS,
            'classifier__multioutputclassifier__estimator__C': [0.1, 1.0, 50.0],
        },
        error_score='raise',
    ),
    'Gradient Boosting': BettorGridSearchCV(
        estimator=ClassifierBettor(
            classifier=make_pipeline(
                make_column_transformer(
                    (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
                    remainder='passthrough',
                    sparse_threshold=0,
                    force_int_remainder_cols=False,
                ),
                SimpleImputer(),
                MultiOutputClassifier(HistGradientBoostingClassifier(random_state=10)),
            ),
        ),
        param_grid={
            'betting_markets': BETTING_MARKETS,
            'classifier__multioutputclassifier__estimator__max_depth': [3, 5, 8],
        },
        error_score='raise',
    ),
}
DEFAULT_PARAM_CHECKED = {
    'leagues': [
        '"England"',
        '"Spain"',
        '"France"',
    ],
    'years': [
        '2020',
        '2021',
        '2022',
        '2023',
        '2024',
        '2025',
    ],
    'divisions': ['1'],
}
VISIBILITY_LEVELS_DATALOADER_CREATION = {
    'sport': 2,
    'parameters': 3,
    'training_parameters': 4,
    'control': 5,
}
VISIBILITY_LEVELS_DATALOADER_LOADING = {
    'dataloader': 2,
    'control': 3,
}
VISIBILITY_LEVELS_MODEL_CREATION = {
    'model': 2,
    'dataloader': 3,
    'evaluation': 4,
    'control': 5,
}
VISIBILITY_LEVELS_MODEL_LOADING = {
    'dataloader_model': 2,
    'evaluation': 3,
    'control': 4,
}
DELAY = 0.001


nest_asyncio.apply()


class State(rx.State):
    """The index page state."""

    dataloader_error: bool = False
    model_error: bool = False

    # Elements
    visibility_level: int = 1
    loading: bool = False

    # Mode
    mode_category: str = 'Data'
    mode_type: str = 'Create'

    # Message
    streamed_message: str = ''

    @staticmethod
    def process_cols(col: str) -> str:
        """Proces a column."""
        return " ".join([" ".join(token.split('_')).title() for token in col.split('__')])

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """You can create or load a dataloader to grab historical and fixtures
        data. Plus, you can create or load a betting model to test how it performs and find value bets for
        upcoming games.
        <br><br>
        <strong>Data, Create</strong><br>
        Create a new dataloader<br><br>

        <strong>Data, Load</strong><br>
        Load an existing dataloader.<br><br>

        <strong>Modelling, Create</strong><br>
        Create a new beting model.<br><br>

        <strong>Modelling, Load</strong><br>
        Load an existing betting model."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'

        # Message
        self.streamed_message = ''


class DataloaderState(State):
    """The dataloader state."""

    all_leagues: list[list[str]] = []  # noqa: RUF012
    all_years: list[list[str]] = []  # noqa: RUF012
    all_divisions: list[list[str]] = []  # noqa: RUF012
    param_checked: dict[str | int, bool] = {}  # noqa: RUF012
    dataloader_filename: str | None = None
    data_title: str | None = None
    loading_db: bool = False


class DataloaderCreationState(DataloaderState):
    """The dataloader creation state."""

    sport_selection: str = 'Soccer'
    all_params: list[dict[str, Any]] = []  # noqa: RUF012
    leagues: list[str] = []  # noqa: RUF012
    years: list[int] = []  # noqa: RUF012
    divisions: list[int] = []  # noqa: RUF012
    params: list[dict[str, Any]] = []  # noqa: RUF012
    default_param_checked: dict[str, list[str]] = DEFAULT_PARAM_CHECKED
    odds_types: list[str] = []  # noqa: RUF012
    param_grid: list[dict] = []  # noqa: RUF012
    odds_type: str = 'market_average'
    drop_na_thres: list = [0.0]  # noqa: RUF012
    dataloader_serialized: str | None = None
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
    data: list | None = None
    data_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """Begin by selecting your sport. Currently, only soccer is
        available, but more sports will be added soon!"""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @staticmethod
    def process_form_data(form_data: dict[str, str]) -> list[str]:
        """Process the form data."""
        return [key.replace('"', '') for key in form_data]

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(cast(str, self.dataloader_serialized), 'iso8859_16')
        return rx.download(data=dataloader, filename=self.dataloader_filename)

    @rx.event
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

    @rx.event
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

    @rx.event
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

    @rx.event
    def handle_submit_leagues(self: Self, leagues_form_data: dict) -> None:
        """Handle the form submit."""
        self.leagues = self.process_form_data(leagues_form_data)
        self.update_params()

    @rx.event
    def handle_submit_years(self: Self, years_form_data: dict) -> None:
        """Handle the form submit."""
        self.years = [int(year) for year in self.process_form_data(years_form_data)]
        self.update_params()

    @rx.event
    def handle_submit_divisions(self: Self, divisions_form_data: dict) -> None:
        """Handle the form submit."""
        self.divisions = [int(division) for division in self.process_form_data(divisions_form_data)]
        self.update_params()

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['sport']:
            self.dataloader_filename = 'dataloader.pkl'
            self.all_params = DATALOADERS[self.sport_selection].get_all_params()
            self.all_leagues = list(chunked(sorted({params['league'] for params in self.all_params}), 6))
            self.all_years = list(chunked(sorted({params['year'] for params in self.all_params}), 5))
            self.all_divisions = list(chunked(sorted({params['division'] for params in self.all_params}), 1))
            self.leagues = [league.replace('"', '') for league in DEFAULT_PARAM_CHECKED['leagues']]
            self.years = [int(year) for year in DEFAULT_PARAM_CHECKED['years']]
            self.divisions = [int(division) for division in DEFAULT_PARAM_CHECKED['divisions']]
            self.loading = False
            yield
            message = """You can configure the dataloader by selecting the type of training data to include. The
            fixtures data follow the same schema, ensuring consistency for applying machine learning
            models during training and inference.<br><br>

            <strong>Training data</strong><br>
            Choose specific leagues, divisions, and years to include.<br><br>

            <strong>Fixtures data</strong><br>
            The selection of leagues, divisions, and years does not impact the fixtures data, which includes all
            upcoming matches."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['parameters']:
            self.update_params()
            self.param_grid = [{k: [v] for k, v in param.items()} for param in self.params]
            self.odds_types = DATALOADERS[self.sport_selection](self.param_grid).get_odds_types()
            self.loading = False
            yield
            message = """The training data consists of input, output, and odds, while the fixtures include only
            input and odds.<br><br>

            <strong>Training data</strong><br>
            You can choose the type of odds to use. Additionally, you can set a tolerance
            level for missing values in the training data. Columns with missing values exceeding
            this tolerance will be removed.<br><br>

            <strong>Fixtures data</strong><br>
            The selections made for the training data affect the fixtures data because their schema aligns
            with the schema of the training data."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_DATALOADER_CREATION['training_parameters']:
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

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None

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


class DataloaderLoadingState(DataloaderState):
    """The dataloader loading state."""

    odds_type: str | None = None
    drop_na_thres: float | None = None
    dataloader_serialized: str | None = None
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
    data: list | None = None
    data_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """Select a dataloader file to extract the latest training and fixtures data."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """Uploaded file is not a dataloader. Please try again."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """Uploaded file is a dataloader. You may proceed to the next step."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    def download_dataloader(self: Self) -> EventSpec:
        """Download the dataloader."""
        dataloader = bytes(cast(str, self.dataloader_serialized), 'iso8859_16')
        return rx.download(data=dataloader, filename=self.dataloader_filename)

    @rx.event
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

    @rx.event
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

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_DATALOADER_LOADING['dataloader']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
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
                **dict.fromkeys({params['year'] for params in dataloader.param_grid_}, True),
                **dict.fromkeys({params['division'] for params in dataloader.param_grid_}, True),
            }
            self.odds_type = dataloader.odds_type_
            self.drop_na_thres = dataloader.drop_na_thres_
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

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


class ModelCreationState(State):
    """The model creation state."""

    model_selection: str = 'Odds Comparison'
    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    model_serialized: str | None = None
    model_filename: str | None = None
    evaluation_selection: str = 'Backtesting'
    backtesting_results: list | None = None
    backtesting_results_cols: list | None = None
    optimal_params: list | None = None
    optimal_params_cols: list | None = None
    value_bets: list | None = None
    value_bets_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """Begin by selecting a betting model. Currently, three options are available.<br><br>

        <strong>Odds Comparison Model</strong><br>
        Calculates probabilities based on average odds and identifies value bets.<br><br>

        <strong>Logistic Regression Model</strong><br>
        Fits a logistic regression classifier to the training data with various
        hyperparameters, managing both categorical and missing values.<br><br>

        <strong>Gradient Boosting Model</strong><br>
        Fits a gradient boosting classifier to the training data with various
        hyperparameters, also handling categorical and missing values."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    def download_model(self: Self) -> EventSpec:
        """Download the model."""
        model = bytes(cast(str, self.model_serialized), 'iso8859_16')
        return rx.download(data=model, filename=self.model_filename)

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """Uploaded file is not a dataloader. Please try again."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """Uploaded file is a dataloader. You may proceed to the next step."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['model']:
            self.loading = False
            yield
            message = (
                """Upload a dataloader to use with the model for backtesting its performance or finding value bets."""
            )
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['dataloader']:
            self.loading = False
            yield
            message = """Choose whether to backtest the model or predict value bets.<br><br>

            Backtesting uses 3-fold time ordered cross-validation with a constant betting
            stake of 50 and an initial cash balance of 10000. After backtesting, the
            model is fitted to the entire training set.<br><br>

            The model can also predict value bets using the fixtures data. The model is fitted
            to the entire training set before making predictions."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_CREATION['evaluation']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            model = MODELS[self.model_selection]
            model.fit(X_train, Y_train, O_train)
            self.model_serialized = str(cloudpickle.dumps(model), 'iso8859_16')
            self.model_filename = 'model.pkl'
            if self.evaluation_selection == 'Backtesting':
                backtesting_results = backtest(model, X_train, Y_train, O_train, cv=TimeSeriesSplit(3)).reset_index()
                self.backtesting_results = backtesting_results.fillna('NA').to_dict('records')
                self.backtesting_results_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col))
                    for col in backtesting_results.columns
                ]
                self.optimal_params = [
                    {'Parameter name': name, 'Optimal value': value} for name, value in model.best_params_.items()
                ]
                self.optimal_params_cols = [
                    ag_grid.column_def(field='Parameter name'),
                    ag_grid.column_def(field='Optimal value'),
                ]
            elif self.evaluation_selection == 'Value bets':
                X_fix, *_ = dataloader.extract_fixtures_data()
                value_bets = pd.DataFrame(np.round(1 / model.predict_proba(X_fix), 2), columns=model.betting_markets_)
                value_bets = pd.concat(
                    [X_fix.reset_index()[['date', 'league', 'division', 'home_team', 'away_team']], value_bets],
                    axis=1,
                )
                self.value_bets = value_bets.fillna('NA').to_dict('records')
                self.value_bets_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in value_bets.columns
                ]
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'

        # Model
        self.model_selection = 'Odds Comparison'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None

        # Evaluation
        self.model_serialized = None
        self.model_filename = None
        self.evaluation_selection = 'Backtesting'
        self.backtesting_results = None
        self.backtesting_results_cols = None
        self.optimal_params = None
        self.optimal_params_cols = None
        self.value_bets = None
        self.value_bets_cols = None

        # Message
        self.streamed_message = ''


class ModelLoadingState(State):
    """The model loading state."""

    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    model_serialized: str | None = None
    model_filename: str | None = None
    evaluation_selection: str = 'Backtesting'
    backtesting_results: list | None = None
    backtesting_results_cols: list | None = None
    optimal_params: list | None = None
    optimal_params_cols: list | None = None
    value_bets: list | None = None
    value_bets_cols: list | None = None

    async def on_load(self: Self) -> AsyncGenerator:
        """Event on page load."""
        message = """Upload a dataloader and a betting model to backtest performance or identify value bets."""
        self.streamed_message = ''
        for char in message:
            await asyncio.sleep(DELAY)
            self.streamed_message += char
            yield

    @rx.event
    def download_model(self: Self) -> EventSpec:
        """Download the model."""
        model = bytes(cast(str, self.model_serialized), 'iso8859_16')
        return rx.download(data=model, filename=self.model_filename)

    @rx.event
    async def handle_dataloader_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of dataloader files."""
        self.loading = True
        yield
        for file in files:
            dataloader = await file.read()
            self.dataloader_serialized = str(dataloader, 'iso8859_16')
            self.dataloader_filename = Path(file.filename).name
        self.loading = False
        yield
        dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
        if not isinstance(dataloader, BaseDataLoader):
            self.dataloader_error = True
            message = """Uploaded file is not a dataloader. Please try again."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.dataloader_error = False
            message = """Uploaded file is a dataloader. You may proceed to the next step."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def handle_model_upload(self: Self, files: list[rx.UploadFile]) -> AsyncGenerator:
        """Handle the upload of model files."""
        self.loading = True
        yield
        for file in files:
            model = await file.read()
            self.model_serialized = str(model, 'iso8859_16')
            self.model_filename = Path(file.filename).name
        self.loading = False
        yield
        model = cloudpickle.loads(bytes(cast(str, self.model_serialized), 'iso8859_16'))
        if not isinstance(model, BaseBettor):
            self.model_error = True
            message = """Uploaded file is not a betting model. Please try again."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        else:
            self.model_error = False
            message = """Uploaded file is a betting model. You may proceed to the next step."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield

    @rx.event
    async def submit_state(self: Self) -> AsyncGenerator:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == VISIBILITY_LEVELS_MODEL_LOADING['dataloader_model']:
            self.loading = False
            yield
            message = """Choose whether to backtest the model or predict value bets.<br><br>

            Backtesting uses 3-fold time-ordered cross-validation with a constant betting
            stake of 50 and an initial cash balance of 10000. After backtesting, the
            model is fitted to the entire training set.<br><br>

            The model can also predict value bets using the fixtures data. The model is
            fitted to the entire training set before making predictions."""
            self.streamed_message = ''
            for char in message:
                await asyncio.sleep(DELAY)
                self.streamed_message += char
                yield
        elif self.visibility_level == VISIBILITY_LEVELS_MODEL_LOADING['evaluation']:
            dataloader = cloudpickle.loads(bytes(cast(str, self.dataloader_serialized), 'iso8859_16'))
            if hasattr(dataloader, 'odds_type_') and hasattr(dataloader, 'drop_na_thres_'):
                X_train, Y_train, O_train = dataloader.extract_train_data(
                    odds_type=dataloader.odds_type_,
                    drop_na_thres=dataloader.drop_na_thres_,
                )
            else:
                X_train, Y_train, O_train = dataloader.extract_train_data()
            model = cloudpickle.loads(bytes(cast(str, self.model_serialized), 'iso8859_16'))
            model.fit(X_train, Y_train, O_train)
            self.model_serialized = str(cloudpickle.dumps(model), 'iso8859_16')
            self.model_filename = 'model.pkl'
            if self.evaluation_selection == 'Backtesting':
                backtesting_results = backtest(model, X_train, Y_train, O_train, cv=TimeSeriesSplit(3)).reset_index()
                self.backtesting_results = backtesting_results.fillna('NA').to_dict('records')
                self.backtesting_results_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col))
                    for col in backtesting_results.columns
                ]
                self.optimal_params = [
                    {'Parameter name': name, 'Optimal value': value} for name, value in model.best_params_.items()
                ]
                self.optimal_params_cols = [
                    ag_grid.column_def(field='Parameter name'),
                    ag_grid.column_def(field='Optimal value'),
                ]
            elif self.evaluation_selection == 'Value bets':
                X_fix, *_ = dataloader.extract_fixtures_data()
                value_bets = pd.DataFrame(np.round(1 / model.predict_proba(X_fix), 2), columns=model.betting_markets_)
                value_bets = pd.concat(
                    [X_fix.reset_index()[['date', 'league', 'division', 'home_team', 'away_team']], value_bets],
                    axis=1,
                )
                self.value_bets = value_bets.fillna('NA').to_dict('records')
                self.value_bets_cols = [
                    ag_grid.column_def(field=col, header_name=self.process_cols(col)) for col in value_bets.columns
                ]
            self.loading = False
            yield
        self.visibility_level += 1

    @rx.event
    def reset_state(self: Self) -> None:
        """Reset handler."""

        self.dataloader_error = False
        self.model_error = False

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'

        # Data
        self.dataloader_serialized = None
        self.dataloader_filename = None

        # Evaluation
        self.model_serialized = None
        self.model_filename = None
        self.evaluation_selection = 'Backtesting'
        self.backtesting_results = None
        self.backtesting_results_cols = None
        self.optimal_params = None
        self.optimal_params_cols = None
        self.value_bets = None
        self.value_bets_cols = None

        # Message
        self.streamed_message = ''
