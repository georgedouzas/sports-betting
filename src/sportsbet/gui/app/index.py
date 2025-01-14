"""Index page."""

import reflex as rx

from .components import bot, control, mode, navbar, save_dataloader, save_model, sidebar
from .states import (
    DataloaderCreationState,
    DataloaderLoadingState,
    ModelCreationState,
    ModelLoadingState,
    State,
)


@rx.page(route="/", on_load=State.on_load)
def index() -> rx.Component:
    """Index page."""
    return rx.box(
        navbar(),
        rx.hstack(
            rx.vstack(
                rx.cond(
                    (State.mode_category == 'Data') & (State.mode_type == 'Create'),
                    sidebar(
                        mode(State, 'Create a dataloader'),
                        control=control(State, False, save_dataloader(DataloaderCreationState), '/dataloader/creation'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == 'Data') & (State.mode_type == 'Load'),
                    sidebar(
                        mode(State, 'Load a dataloader'),
                        control=control(State, False, save_dataloader(DataloaderLoadingState), '/dataloader/loading'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == 'Modelling') & (State.mode_type == 'Create'),
                    sidebar(
                        mode(State, 'Create a model'),
                        control=control(State, False, save_model(ModelCreationState), '/model/creation'),
                    ),
                ),
                rx.cond(
                    (State.mode_category == 'Modelling') & (State.mode_type == 'Load'),
                    sidebar(
                        mode(State, 'Load a model'),
                        control=control(State, False, save_model(ModelLoadingState), '/model/loading'),
                    ),
                ),
            ),
            bot(State, 2),
            padding='2%',
        ),
    )
