"""Model creation page."""

from typing import cast

import reflex as rx

from .components import SIDEBAR_OPTIONS, bot, dataloader, mode, model_data, navbar, save_model, submit_reset, title
from .states import VISIBILITY_LEVELS_MODEL_CREATION as VL
from .states import ModelCreationState


def model(state: rx.State) -> rx.Component:
    """The model component."""
    return rx.vstack(
        title('Model', 'wand'),
        rx.text('Select a model', size='1'),
        rx.select(
            items=['Odds Comparison', 'Logistic Regression', 'Gradient Boosting'],
            value=state.model_selection,
            disabled=state.visibility_level > VL['model'],
            on_change=state.set_model_selection,
            width='160px',
        ),
        margin_top='10px',
    )


def run(state: rx.State) -> rx.Component:
    """The usage component."""
    return rx.cond(
        state.visibility_level > VL['dataloader'],
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


@rx.page(route="/model/creation", on_load=ModelCreationState.on_load)
def model_creation_page() -> rx.Component:
    """Main page."""
    return rx.box(
        navbar(),
        rx.vstack(
            mode(ModelCreationState, 'Create a model'),
            model(ModelCreationState),
            dataloader(ModelCreationState, VL['model']),
            run(ModelCreationState),
            submit_reset(
                ModelCreationState,
                (
                    (~cast(rx.Var, ModelCreationState.dataloader_serialized).bool())
                    & (ModelCreationState.visibility_level == VL['dataloader'])
                )
                | (ModelCreationState.visibility_level > VL['evaluation']),
            ),
            save_model(ModelCreationState, VL['evaluation']),
            **SIDEBAR_OPTIONS,
        ),
        model_data(ModelCreationState, VL['control']),
        bot(ModelCreationState, VL['control']),
    )
