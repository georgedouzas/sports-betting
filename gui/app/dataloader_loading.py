"""Index page."""

from itertools import batched
from pathlib import Path
from typing import Self

import cloudpickle
import nest_asyncio
import reflex as rx

from sportsbet.datasets import SoccerDataLoader

from .components.dataloader_loading import main
from .index import State

DATALOADERS = {
    'Soccer': SoccerDataLoader,
}
DEFAULT_STATE_VALS = {
    'mode': {
        'category': 'Data',
        'type': 'Create',
    },
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

    def submit_state(self: Self) -> None:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        elif self.visibility_level == 2:
            dataloader = cloudpickle.loads(bytes(self.dataloader_serialized, 'iso8859_16'))
            all_params = dataloader.get_all_params()
            self.all_leagues = list(batched(sorted({params['league'] for params in all_params}), 6))
            self.all_years = list(batched(sorted({params['year'] for params in all_params}), 5))
            self.all_divisions = list(batched(sorted({params['division'] for params in all_params}), 1))
            self.loading = False
            self.param_checked = {
                **{f'"{key}"': True for key in {params['league'] for params in dataloader.param_grid_}},
                **{key: True for key in {params['year'] for params in dataloader.param_grid_}},
                **{key: True for key in {params['division'] for params in dataloader.param_grid_}},
            }
            self.odds_type = dataloader.odds_type_
            self.drop_na_thres = dataloader.drop_na_thres_
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


@rx.page(route="/dataloader/loading")
def dataloader_loading_page() -> rx.Component:
    """Main page."""
    return main(DataloaderLoadingState)
