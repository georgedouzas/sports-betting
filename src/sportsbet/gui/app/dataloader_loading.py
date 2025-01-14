"""Dataloader loading page."""

from typing import cast

import reflex as rx

from .components import bot, control, dataloader, dataloader_data, mode, navbar, save_dataloader, sidebar, title
from .states import VISIBILITY_LEVELS_DATALOADER_LOADING as VL
from .states import DataloaderLoadingState


def parameters(state: rx.State) -> rx.Component:
    """The parameters component."""

    def _checkboxes(row: list[str], state: rx.State) -> rx.Component:
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

    def _dialog(name: str, icon_name: str, rows: list[list[str]]) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    rx.tooltip(rx.icon(icon_name), content=name),
                    size='4',
                    variant='outline',
                    disabled=state.visibility_level > VL['control'],
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
                    rx.hstack(rx.foreach(rows, lambda row: _checkboxes(row, state))),
                    width="100%",
                ),
            ),
        )

    return rx.cond(
        state.visibility_level > VL['dataloader'],
        rx.vstack(
            title('Parameters', 'proportions'),
            rx.hstack(
                _dialog('Leagues', 'earth', state.all_leagues),
                _dialog('Years', 'calendar', state.all_years),
                _dialog('Divisions', 'gauge', state.all_divisions),
            ),
            rx.text(f'Odds type: {state.odds_type}', size='1'),
            rx.text(f'Drop NA threshold of columns: {state.drop_na_thres}', size='1'),
            margin_top='10px',
        ),
    )


@rx.page(route="/dataloader/loading", on_load=DataloaderLoadingState.on_load)
def dataloader_loading_page() -> rx.Component:
    """Main page."""
    return rx.box(
        navbar(),
        rx.hstack(
            sidebar(
                mode(DataloaderLoadingState, 'Load a dataloader'),
                dataloader(DataloaderLoadingState, 1),
                parameters(DataloaderLoadingState),
                control=control(
                    DataloaderLoadingState,
                    (~cast(rx.Var, DataloaderLoadingState.dataloader_serialized).bool())
                    | (DataloaderLoadingState.visibility_level > VL['dataloader']),
                    save=save_dataloader(DataloaderLoadingState, VL['control']),
                ),
            ),
            dataloader_data(DataloaderLoadingState, VL['control']),
            bot(DataloaderLoadingState, VL['control']),
            padding='2%',
        ),
        height='1100px',
    )
