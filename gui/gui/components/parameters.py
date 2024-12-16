"""Page parameters components."""

from collections.abc import Callable

import reflex as rx


def checkboxes(row: list[str], state: rx.State) -> rx.Component:
    """Checkbox of parameter value."""
    return rx.vstack(
        rx.foreach(
            row,
            lambda name: rx.checkbox(
                name,
                default_checked=state.default_param_checked[name.to_string()],
                checked=state.param_checked[name.to_string()],
                name=name.to_string(),
                on_change=lambda checked: state.update_param_checked(name, checked),
            ),
        ),
    )


def dialog(name: str, icon_name: str, state: rx.State) -> Callable:
    """Dialog component."""

    def _dialog(rows: list[list[str]], on_submit: Callable) -> rx.Component:
        """The dialog component."""
        return rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    rx.tooltip(rx.icon(icon_name), content=name),
                    size='4',
                    variant='outline',
                    disabled=state.parameters_disabled,
                )
            ),
            rx.dialog.content(
                rx.form.root(
                    rx.dialog.title(name),
                    rx.dialog.description(
                        f'Select the {name.lower()} to include in the training data.',
                        size="2",
                        margin_bottom="16px",
                    ),
                    rx.hstack(rx.foreach(rows, lambda row: checkboxes(row, state))),
                    rx.flex(
                        rx.dialog.close(rx.button('Submit', type='submit')),
                        justify='end',
                        spacing="3",
                        margin_top="50px",
                    ),
                    on_submit=on_submit,
                    reset_on_submit=False,
                    width="100%",
                ),
            ),
        )

    return _dialog


def parameters_selection(state: rx.State) -> rx.Component:
    """The parameters title."""
    return rx.vstack(
        rx.text('Leagues, years and divisions selection', size='1', hidden=state.parameters_disabled),
        rx.hstack(
            dialog('Leagues', 'earth', state)(state.all_leagues, state.handle_submit_leagues),
            dialog('Years', 'calendar', state)(state.all_years, state.handle_submit_years),
            dialog('Divisions', 'gauge', state)(state.all_divisions, state.handle_submit_divisions),
        ),
    )
