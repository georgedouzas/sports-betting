"""Model creation page."""

from typing import cast

import reflex as rx

from .components import (
    SIDEBAR_OPTIONS,
    bot,
    dataloader,
    mode,
    model,
    model_data,
    navbar,
    save_model,
    submit_reset,
    title,
)
from .states import VISIBILITY_LEVELS_MODEL_LOADING as VL
from .states import ModelLoadingState


def run(state: rx.State) -> rx.Component:
    """The usage component."""
    return rx.cond(
        state.visibility_level > VL['dataloader_model'],
        rx.vstack(
            title('Run', 'play'),
            rx.text('Run the model', size='1'),
            rx.select(
                items=['Backtesting', 'Value bets'],
                value=state.evaluation_selection,
                disabled=state.visibility_level > VL['evaluation'],
                on_change=state.set_evaluation_selection,
                width='160px',
            ),
            margin_top='10px',
        ),
    )


@rx.page(route="/model/loading", on_load=ModelLoadingState.on_load)
def model_loading_page() -> rx.Component:
    """Main page."""
    return rx.container(
        navbar(),
        rx.vstack(
            mode(ModelLoadingState, 'Load a model'),
            dataloader(ModelLoadingState, 1),
            model(ModelLoadingState, 1),
            run(ModelLoadingState),
            submit_reset(
                ModelLoadingState,
                (~cast(rx.Var, ModelLoadingState.dataloader_serialized).bool())
                | (~cast(rx.Var, ModelLoadingState.model_serialized).bool())
                | (ModelLoadingState.visibility_level > VL['evaluation']),
            ),
            save_model(ModelLoadingState, VL['evaluation']),
            **SIDEBAR_OPTIONS,
        ),
        model_data(ModelLoadingState, VL['control']),
        bot(ModelLoadingState, VL['control']),
    )