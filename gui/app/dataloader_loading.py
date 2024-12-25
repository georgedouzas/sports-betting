"""Index page."""

from collections.abc import Callable
from itertools import batched
from pathlib import Path
from typing import Self

import cloudpickle
import nest_asyncio
import reflex as rx
from reflex.event import EventSpec
from reflex_ag_grid import ag_grid

from sportsbet.datasets import SoccerDataLoader

from .components import SIDEBAR_OPTIONS, control_buttons, home, select_mode, title
from .index import State

DATALOADERS = {
    'Soccer': SoccerDataLoader,
}

nest_asyncio.apply()


class DataloaderLoadingState(State):
    """The toolbox state."""

    # Data
    dataloader_serialized: str | None = None
    dataloader_filename: str | None = None
    all_leagues: list[list[str]] = []
    all_years: list[list[str]] = []
    all_divisions: list[list[str]] = []
    param_checked: dict[str, bool] = {}
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

    @rx.event
    async def handle_upload(self: Self, files: list[rx.UploadFile]) -> None:
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
        return rx.download(data=dataloader, filename='dataloader.pkl')

    @staticmethod
    def process_cols(col: str) -> str:
        """Proces a column."""
        return " ".join([" ".join(token.split('_')).title() for token in col.split('__')])

    def submit_state(self: Self) -> None:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        elif self.visibility_level == 2:
            dataloader = cloudpickle.loads(bytes(self.dataloader_serialized, 'iso8859_16'))
            X_train, Y_train, O_train = dataloader.extract_train_data(
                odds_type=dataloader.odds_type_,
                drop_na_thres=dataloader.drop_na_thres_,
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
            all_params = dataloader.get_all_params()
            self.all_leagues = list(batched(sorted({params['league'] for params in all_params}), 6))
            self.all_years = list(batched(sorted({params['year'] for params in all_params}), 5))
            self.all_divisions = list(batched(sorted({params['division'] for params in all_params}), 1))
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


def checkboxes(row: list[str], state: rx.State) -> rx.Component:
    """Checkbox of parameter value."""

    return rx.vstack(
        rx.foreach(
            row,
            lambda name: rx.checkbox(
                name,
                disabled=True,
                default_checked=state.param_checked[name.to_string()],
                name=name.to_string(),
            ),
        ),
    )


def dialog(name: str, icon_name: str, state: rx.State) -> Callable:
    """Dialog component."""

    def _dialog(rows: list[list[str]]) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    rx.tooltip(rx.icon(icon_name), content=name),
                    size='4',
                    variant='outline',
                    disabled=state.visibility_level > 3,
                ),
            ),
            rx.dialog.content(
                rx.form.root(
                    rx.dialog.title(name),
                    rx.dialog.description(
                        f'{name} included in the training data.',
                        size="2",
                        margin_bottom="16px",
                    ),
                    rx.hstack(rx.foreach(rows, lambda row: checkboxes(row, state))),
                    width="100%",
                ),
            ),
        )

    return _dialog


@rx.page(route="/dataloader/loading")
def dataloader_loading_page() -> rx.Component:
    """Main page."""
    return rx.container(
        rx.vstack(
            home(),
            rx.divider(),
            # Mode selection
            title('Mode', 'blend'),
            select_mode(DataloaderLoadingState, 'Load a dataloader'),
            # Dataloader selection
            rx.cond(
                DataloaderLoadingState.visibility_level > 1,
                title('Dataloader', 'database'),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level > 1,
                rx.upload(
                    rx.vstack(
                        rx.button(
                            'Select File',
                            bg='white',
                            color='rgb(107,99,246)',
                            border=f'1px solid rgb(107,99,246)',
                            disabled=DataloaderLoadingState.dataloader_serialized.bool(),
                        ),
                        rx.text('Drag and drop', size='2'),
                    ),
                    id='dataloader',
                    multiple=False,
                    no_keyboard=True,
                    no_drag=DataloaderLoadingState.dataloader_serialized.bool(),
                    on_drop=DataloaderLoadingState.handle_upload(rx.upload_files(upload_id='dataloader')),
                    border='1px dotted blue',
                    padding='35px',
                ),
            ),
            rx.cond(
                DataloaderLoadingState.dataloader_serialized,
                rx.text(f'Dataloader: {DataloaderLoadingState.dataloader_filename}', size='1'),
            ),
            # Parameters presentation
            rx.cond(
                DataloaderLoadingState.visibility_level > 2,
                title('Parameters', 'proportions'),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level > 2,
                rx.hstack(
                    dialog('Leagues', 'earth', DataloaderLoadingState)(DataloaderLoadingState.all_leagues),
                    dialog('Years', 'calendar', DataloaderLoadingState)(DataloaderLoadingState.all_years),
                    dialog('Divisions', 'gauge', DataloaderLoadingState)(DataloaderLoadingState.all_divisions),
                ),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level > 2,
                rx.text(f'Odds type: {DataloaderLoadingState.odds_type}', size='1'),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level > 2,
                rx.text(f'Drop NA threshold of columns: {DataloaderLoadingState.drop_na_thres}', size='1'),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level > 2,
                rx.button(
                    'Save',
                    position='fixed',
                    top='620px',
                    left='275px',
                    width='70px',
                    on_click=DataloaderLoadingState.download_dataloader,
                ),
            ),
            # Control
            control_buttons(
                DataloaderLoadingState,
                (~DataloaderLoadingState.dataloader_serialized.bool()) | (DataloaderLoadingState.visibility_level > 2),
            ),
            **SIDEBAR_OPTIONS,
        ),
        rx.vstack(
            rx.cond(
                DataloaderLoadingState.visibility_level == 3,
                rx.hstack(
                    rx.heading(
                        'Training data', size='7', position='fixed', left='450px', top='50px', color_scheme='blue'
                    )
                ),
            ),
            rx.hstack(
                rx.vstack(
                    rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Input')),
                    rx.cond(
                        DataloaderLoadingState.visibility_level == 3,
                        ag_grid(
                            id='X_train',
                            row_data=DataloaderLoadingState.X_train,
                            column_defs=DataloaderLoadingState.X_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Output')),
                    rx.cond(
                        DataloaderLoadingState.visibility_level == 3,
                        ag_grid(
                            id='Y_train',
                            row_data=DataloaderLoadingState.Y_train,
                            column_defs=DataloaderLoadingState.Y_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Odds')),
                    rx.cond(
                        DataloaderLoadingState.visibility_level == 3,
                        ag_grid(
                            id='O_train',
                            row_data=DataloaderLoadingState.O_train,
                            column_defs=DataloaderLoadingState.O_train_cols,
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
                DataloaderLoadingState.visibility_level == 3,
                rx.hstack(
                    rx.heading(
                        'Fixtures data', size='7', position='fixed', left='450px', top='370px', color_scheme='blue'
                    )
                ),
            ),
            rx.cond(
                DataloaderLoadingState.visibility_level == 3,
                rx.cond(
                    DataloaderLoadingState.X_fix,
                    rx.hstack(
                        rx.vstack(
                            rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Input')),
                            rx.cond(
                                DataloaderLoadingState.visibility_level == 3,
                                ag_grid(
                                    id='X_fix',
                                    row_data=DataloaderLoadingState.X_fix,
                                    column_defs=DataloaderLoadingState.X_fix_cols,
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        rx.vstack(
                            rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Output')),
                            rx.cond(
                                DataloaderLoadingState.visibility_level == 3,
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
                            rx.cond(DataloaderLoadingState.visibility_level == 3, rx.heading('Odds')),
                            rx.cond(
                                DataloaderLoadingState.visibility_level == 3,
                                ag_grid(
                                    id='O_fix',
                                    row_data=DataloaderLoadingState.O_fix,
                                    column_defs=DataloaderLoadingState.O_fix_cols,
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
    )
