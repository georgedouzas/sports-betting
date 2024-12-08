"""Selection of the dataloader's parameters."""

from typing import Self

import reflex as rx

from ...template import template
from .params import ParamsState


class TrainingState(ParamsState):
    """The training state."""

    odds_type: str | None = None
    drop_na_thres: float | None = None

    def reset_training_state(self: Self) -> None:
        """Reset the parameters state."""
        self.reset_params_state()
        self.odds_type = None
        self.drop_na_thres = None


def main_component() -> rx.Component:
    """The main component."""
    return rx.vstack(
        rx.select(
            TrainingState.odds_types,
            placeholder='Selection of odds type',
        ),
        rx.vstack(
            rx.text('Selection of NA columns threshold', size='1'),
            rx.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                default_value=0.0,
                placeholder='Selection of tool',
            ),
            style={'margin-top': '15px'},
        ),
    )


def control_component() -> rx.Component:
    """The control component."""
    return rx.hstack(
        rx.cond(
            ParamsState.params,
            rx.link(rx.icon_button('check'), href='train-data-selection'),
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
            rx.icon_button('arrow-left', on_click=TrainingState.reset_training_state),
            href='/dataloader-params-selection',
        ),
    )


@rx.page(route="/train-data-selection")
@template('Data', 'database', 'Training', 'dumbbell')
def train_data_selection() -> rx.Component:
    """Train data page."""
    return main_component(), control_component()
