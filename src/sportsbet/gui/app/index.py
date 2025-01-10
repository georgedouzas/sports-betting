"""Index page."""

import reflex as rx

from .components import SIDEBAR_OPTIONS, bot, mode, navbar, submit_reset
from .states import State


@rx.page(route="/", on_load=State.on_load)
def index() -> rx.Component:
    """Index page."""
    return rx.box(
        navbar(),
        rx.vstack(
            rx.cond(
                (State.mode_category == 'Data') & (State.mode_type == 'Create'),
                rx.vstack(
                    mode(State, 'Create a dataloader'),
                    submit_reset(State, False, '/dataloader/creation'),
                ),
            ),
            rx.cond(
                (State.mode_category == 'Data') & (State.mode_type == 'Load'),
                rx.vstack(
                    mode(State, 'Load a dataloader'),
                    submit_reset(State, False, '/dataloader/loading'),
                ),
            ),
            rx.cond(
                (State.mode_category == 'Modelling') & (State.mode_type == 'Create'),
                rx.vstack(
                    mode(State, 'Create a model'),
                    submit_reset(State, False, '/model/creation'),
                ),
            ),
            rx.cond(
                (State.mode_category == 'Modelling') & (State.mode_type == 'Load'),
                rx.vstack(
                    mode(State, 'Load a model'),
                    submit_reset(State, False, '/model/loading'),
                ),
            ),
            **SIDEBAR_OPTIONS,
        ),
        bot(State, 2),
    )
