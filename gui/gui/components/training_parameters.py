"""Page training components."""

import reflex as rx


def training_parameters_selection(state: rx.State) -> rx.Component:
    """The trianing parameters selection component."""
    return rx.vstack(
        rx.vstack(
            rx.text('Selection of odds type', size='1', hidden=state.training_disabled),
            rx.select(
                state.odds_types,
                default_value=state.odds_types[0],
                on_change=state.handle_odds_type,
                disabled=state.training_disabled,
                width='100%',
            ),
        ),
        rx.vstack(
            rx.text('Selection of NA columns threshold', size='1', hidden=state.training_disabled),
            rx.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                default_value=0.0,
                on_change=state.handle_drop_na_thres,
                disabled=state.training_disabled,
            ),
            style={
                'margin-top': '15px',
                'width': '100%',
            },
        ),
    )
