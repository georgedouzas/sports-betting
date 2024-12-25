"""Load dataloader components."""

from collections.abc import Callable

import reflex as rx

from .common import SIDEBAR_OPTIONS, control_buttons, home, select_mode, title


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


def main(state: rx.State) -> rx.Component:
    """Main container of UI."""
    return rx.container(
        rx.vstack(
            home(),
            rx.divider(),
            # Mode selection
            title('Mode', 'blend'),
            select_mode(state, 'Load a dataloader'),
            # Dataloader selection
            rx.cond(
                state.visibility_level > 1,
                title('Dataloader', 'database'),
            ),
            rx.cond(
                state.visibility_level > 1,
                rx.upload(
                    rx.vstack(
                        rx.button(
                            'Select File',
                            bg='white',
                            color='rgb(107,99,246)',
                            border=f'1px solid rgb(107,99,246)',
                            disabled=state.dataloader_serialized.bool(),
                        ),
                        rx.text('Drag and drop', size='2'),
                    ),
                    id='dataloader',
                    multiple=False,
                    no_keyboard=True,
                    no_drag=state.dataloader_serialized.bool(),
                    on_drop=state.handle_upload(rx.upload_files(upload_id='dataloader')),
                    border='1px dotted blue',
                    padding='35px',
                ),
            ),
            rx.cond(
                state.dataloader_serialized,
                rx.text(f'Dataloader: {state.dataloader_filename}', size='1'),
            ),
            # Parameters presentation
            rx.cond(
                state.visibility_level > 2,
                title('Parameters', 'proportions'),
            ),
            rx.cond(
                state.visibility_level > 2,
                rx.hstack(
                    dialog('Leagues', 'earth', state)(state.all_leagues),
                    dialog('Years', 'calendar', state)(state.all_years),
                    dialog('Divisions', 'gauge', state)(state.all_divisions),
                ),
            ),
            rx.cond(
                state.visibility_level > 2,
                rx.text(f'Odds type: {state.odds_type}', size='1'),
            ),
            rx.cond(
                state.visibility_level > 2,
                rx.text(f'Drop NA threshold of columns: {state.drop_na_thres}', size='1'),
            ),
            # Control
            control_buttons(state, (~state.dataloader_serialized.bool()) | (state.visibility_level > 2)),
            **SIDEBAR_OPTIONS,
        ),
    )
