"""Create dataloader components."""

from collections.abc import Callable

import reflex as rx
from reflex_ag_grid import ag_grid

from ..common import SIDEBAR_OPTIONS, control_buttons, home, select_mode, title


def checkboxes(row: list[str], state: rx.State) -> rx.Component:
    """Checkbox of parameter value."""

    def _in_leagues(name: str) -> rx.Var:
        return state.default_param_checked['leagues'].contains(name.to_string())

    def _in_years(name: str) -> rx.Var:
        return state.default_param_checked['years'].contains(name.to_string())

    def _in_divisions(name: str) -> rx.Var:
        return state.default_param_checked['divisions'].contains(name.to_string())

    return rx.vstack(
        rx.foreach(
            row,
            lambda name: rx.checkbox(
                name,
                default_checked=rx.cond(
                    _in_leagues(name), True, rx.cond(_in_years(name), True, rx.cond(_in_divisions(name), True, False))
                ),
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
                    disabled=state.visibility_level > 3,
                ),
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


def training_parameters_selection(state: rx.State) -> rx.Component:
    """The training parameters selection component."""
    return rx.vstack(
        rx.vstack(
            rx.text('Odds type', size='1'),
            rx.select(
                state.odds_types,
                default_value=state.odds_types[0],
                on_change=state.handle_odds_type,
                disabled=state.visibility_level > 4,
                width='100%',
            ),
        ),
        rx.vstack(
            rx.text('Drop NA threshold of columns', size='1'),
            rx.slider(
                min=0.0,
                max=1.0,
                step=0.01,
                default_value=0.0,
                on_change=state.handle_drop_na_thres,
                disabled=state.visibility_level > 4,
            ),
            style={
                'margin-top': '15px',
                'width': '100%',
            },
        ),
    )


def parameters_selection(state: rx.State) -> rx.Component:
    """The parameters title."""
    return rx.hstack(
        dialog('Leagues', 'earth', state)(state.all_leagues, state.handle_submit_leagues),
        dialog('Years', 'calendar', state)(state.all_years, state.handle_submit_years),
        dialog('Divisions', 'gauge', state)(state.all_divisions, state.handle_submit_divisions),
    )


def main(state: rx.State) -> rx.Component:
    """Main container of UI."""
    return rx.container(
        rx.vstack(
            home(),
            rx.divider(),
            # Mode selection
            title('Mode', 'blend'),
            select_mode(state, 'Create a dataloader'),
            # Sport selection
            rx.cond(
                state.visibility_level > 1,
                title('Sport', 'medal'),
            ),
            rx.cond(
                state.visibility_level > 1,
                rx.text('Select a sport', size='1'),
            ),
            rx.cond(
                state.visibility_level > 1,
                rx.select(
                    items=['Soccer'],
                    value='Soccer',
                    disabled=state.visibility_level > 2,
                    on_change=state.set_sport_selection,
                    width='120px',
                ),
            ),
            # Parameters selection
            rx.cond(
                state.visibility_level > 2,
                title('Parameters', 'proportions'),
            ),
            rx.cond(
                state.visibility_level > 2,
                rx.text('Select parameters', size='1'),
            ),
            rx.cond(
                state.visibility_level > 2,
                parameters_selection(state),
            ),
            # Training parameters selection
            rx.cond(
                state.visibility_level > 3,
                training_parameters_selection(state),
            ),
            rx.cond(
                state.visibility_level > 4,
                rx.button(
                    'Save',
                    position='fixed',
                    top='620px',
                    left='275px',
                    width='70px',
                    on_click=state.download_dataloader,
                ),
            ),
            # Control
            control_buttons(state, state.visibility_level == 5),
            **SIDEBAR_OPTIONS,
        ),
        rx.vstack(
            rx.cond(
                state.visibility_level == 5,
                rx.hstack(
                    rx.heading(
                        'Training data', size='7', position='fixed', left='450px', top='50px', color_scheme='blue'
                    )
                ),
            ),
            rx.hstack(
                rx.vstack(
                    rx.cond(state.visibility_level == 5, rx.heading('Input')),
                    rx.cond(
                        state.visibility_level == 5,
                        ag_grid(
                            id='X_train',
                            row_data=state.X_train,
                            column_defs=state.X_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(state.visibility_level == 5, rx.heading('Output')),
                    rx.cond(
                        state.visibility_level == 5,
                        ag_grid(
                            id='Y_train',
                            row_data=state.Y_train,
                            column_defs=state.Y_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                rx.vstack(
                    rx.cond(state.visibility_level == 5, rx.heading('Odds')),
                    rx.cond(
                        state.visibility_level == 5,
                        ag_grid(
                            id='O_train',
                            row_data=state.O_train,
                            column_defs=state.O_train_cols,
                            height='200px',
                            width='250px',
                            theme='balham',
                        ),
                    ),
                ),
                position='fixed',
                left='450px',
                top='100px',
            ),
        ),
        rx.vstack(
            rx.cond(
                state.visibility_level == 5,
                rx.hstack(
                    rx.heading(
                        'Fixtures data', size='7', position='fixed', left='450px', top='370px', color_scheme='blue'
                    )
                ),
            ),
            rx.cond(
                state.visibility_level == 5,
                rx.cond(
                    state.X_fix,
                    rx.hstack(
                        rx.vstack(
                            rx.cond(state.visibility_level == 5, rx.heading('Input')),
                            rx.cond(
                                state.visibility_level == 5,
                                ag_grid(
                                    id='X_fix',
                                    row_data=state.X_fix,
                                    column_defs=state.X_fix_cols,
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        rx.vstack(
                            rx.cond(state.visibility_level == 5, rx.heading('Output')),
                            rx.cond(
                                state.visibility_level == 5,
                                ag_grid(
                                    id='Y_fix',
                                    row_data=[],
                                    column_defs=[],
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        rx.vstack(
                            rx.cond(state.visibility_level == 5, rx.heading('Odds')),
                            rx.cond(
                                state.visibility_level == 5,
                                ag_grid(
                                    id='O_fix',
                                    row_data=state.O_fix,
                                    column_defs=state.O_fix_cols,
                                    height='200px',
                                    width='250px',
                                    theme='balham',
                                ),
                            ),
                        ),
                        position='fixed',
                        left='450px',
                        top='420px',
                    ),
                    rx.tooltip(
                        rx.icon('ban', position='fixed', left='450px', top='420px', size=60),
                        content='No fixtures were found. Try again later.',
                    ),
                ),
            ),
        ),
    )
