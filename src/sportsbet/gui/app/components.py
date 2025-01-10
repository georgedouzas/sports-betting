"""Page common components."""

import reflex as rx
from reflex_ag_grid import ag_grid

SIDEBAR_OPTIONS = {
    'spacing': '0',
    'position': 'fixed',
    'left': '50px',
    'top': '100px',
    'padding_x': '1em',
    'padding_y': "1.5em",
    'bg': rx.color('blue', 3),
    'height': '600px',
    'width': '20em',
}


def navbar() -> rx.Component:
    """The navigation bar component."""
    return rx.box(
        rx.center(
            rx.link(
                rx.heading("Sports Betting", size="8", weight="bold", color='black'),
                href='https://github.com/georgedouzas/sports-betting',
                underline='none',
                is_external=True,
            ),
            padding_top='5px',
        ),
        bg=rx.color('blue', 3),
        height='60px',
    )


def home() -> rx.Component:
    """Home title."""
    return rx.vstack(
        rx.text('Sports Betting', size='4', weight='bold'),
        rx.divider(width='18em'),
    )


def title(name: str, icon_name: str) -> rx.Component:
    """The title component."""
    return rx.hstack(
        rx.icon(icon_name),
        rx.text(name, size="4"),
        width="100%",
        color=rx.color("accent", 11),
    )


def mode(state: rx.State, content: str) -> rx.Component:
    """Mode component."""
    return rx.vstack(
        title('Mode', 'blend'),
        rx.text(content, size='1'),
        rx.hstack(
            rx.select(
                items=['Data', 'Modelling'],
                value=state.mode_category,
                disabled=state.visibility_level > 1,
                width='120px',
                on_change=state.set_mode_category,
            ),
            rx.select(
                ['Create', 'Load'],
                value=state.mode_type,
                disabled=state.visibility_level > 1,
                width='120px',
                on_change=state.set_mode_type,
            ),
        ),
    )


def dataloader(state: rx.State, visibility_level: int) -> rx.Component:
    """The dataloader component."""
    return rx.vstack(
        rx.cond(
            state.visibility_level > visibility_level,
            rx.vstack(
                title('Dataloader', 'database'),
                rx.upload(
                    rx.button(
                        'Select File',
                        bg='white',
                        color='rgb(107,99,246)',
                        border='1px solid rgb(107,99,246)',
                        disabled=state.dataloader_serialized.bool() & ~state.dataloader_error,
                    ),
                    id='dataloader',
                    multiple=False,
                    no_keyboard=True,
                    no_drag=True,
                    on_drop=state.handle_dataloader_upload(rx.upload_files(upload_id='dataloader')),
                    padding='0px',
                    border=None,
                ),
                margin_top='10px',
            ),
        ),
        rx.cond(
            state.dataloader_serialized & ~state.dataloader_error,
            rx.text(f'Dataloader: {state.dataloader_filename}', size='1'),
        ),
    )


def model(state: rx.State, visibility_level: int) -> rx.Component:
    """The model component."""
    return rx.vstack(
        rx.cond(
            state.visibility_level > visibility_level,
            rx.vstack(
                title('Model', 'wand'),
                rx.upload(
                    rx.button(
                        'Select File',
                        bg='white',
                        color='rgb(107,99,246)',
                        border='1px solid rgb(107,99,246)',
                        disabled=state.model_serialized.bool() & ~state.model_error,
                    ),
                    id='model',
                    multiple=False,
                    no_keyboard=True,
                    no_drag=True,
                    on_drop=state.handle_model_upload(rx.upload_files(upload_id='model')),
                    padding='0px',
                    border=None,
                ),
                margin_top='10px',
            ),
        ),
        rx.cond(
            state.model_serialized & ~state.model_error,
            rx.text(f'Model: {state.model_filename}', size='1'),
        ),
    )


def submit_reset(state: rx.State, disabled: bool, url: str | None = None) -> rx.Component:
    """Submit and reset buttons of UI."""
    return rx.vstack(
        rx.divider(top='630px', position='fixed', width='18em'),
        rx.hstack(
            rx.link(
                rx.button(
                    'Submit',
                    on_click=state.submit_state,
                    disabled=disabled | state.dataloader_error,
                    loading=state.loading,
                    position='fixed',
                    top='650px',
                    width='70px',
                ),
                href=url,
            ),
            rx.link(
                rx.button(
                    'Reset',
                    on_click=state.reset_state,
                    position='fixed',
                    top='650px',
                    left='150px',
                    width='70px',
                ),
                href='/',
            ),
        ),
    )


def save_dataloader(state: rx.State, visibility_level: int) -> rx.Component:
    """The save component."""
    return rx.cond(
        state.visibility_level > visibility_level,
        rx.button(
            'Save',
            position='fixed',
            top='650px',
            left='275px',
            width='70px',
            on_click=state.download_dataloader,
        ),
    )


def save_model(state: rx.State, visibility_level: int) -> rx.Component:
    """The save component."""
    return rx.cond(
        state.visibility_level > visibility_level,
        rx.button(
            'Save',
            position='fixed',
            top='650px',
            left='275px',
            width='70px',
            on_click=state.download_model,
        ),
    )


def dataloader_data(state: rx.State, visibility_level: int) -> rx.Component:
    """Data component of UI."""
    return rx.cond(
        state.visibility_level == visibility_level,
        rx.vstack(
            rx.heading(state.data_title),
            rx.vstack(
                rx.divider(width='900px'),
                ag_grid(
                    id='data',
                    row_data=state.data,
                    column_defs=state.data_cols,
                    height='470px',
                    width='900px',
                    theme='balham',
                ),
                rx.divider(width='900px'),
            ),
            rx.hstack(
                rx.cond(
                    state.X_fix,
                    rx.tooltip(
                        rx.button(
                            rx.icon('database'),
                            on_click=state.switch_displayed_data_category,
                            variant='surface',
                            loading=state.loading_db,
                        ),
                        content='Switch between training or fixtures data',
                    ),
                    rx.tooltip(
                        rx.button(rx.icon('database'), variant='surface', disabled=True),
                        content='Fixtures are not currently available',
                    ),
                ),
                rx.tooltip(
                    rx.button(
                        rx.icon('arrow-up-down'),
                        on_click=state.switch_displayed_data_type,
                        variant='surface',
                        loading=state.loading_db,
                    ),
                    content='Switch between input, output or odds data',
                ),
                margin_top='10px',
            ),
            position='fixed',
            left='400px',
            top='90px',
        ),
    )


def model_data(state: rx.State, visibility_level: int) -> rx.Component:
    """Data component of UI."""
    return rx.cond(
        state.visibility_level == visibility_level,
        rx.cond(
            state.evaluation_selection == 'Backtesting',
            rx.vstack(
                rx.heading('Backtesting results'),
                rx.vstack(
                    rx.divider(width='900px'),
                    ag_grid(
                        id='backtesting',
                        row_data=state.backtesting_results,
                        column_defs=state.backtesting_results_cols,
                        height='225px',
                        width='900px',
                        theme='balham',
                    ),
                    ag_grid(
                        id='parameters',
                        row_data=state.optimal_params,
                        column_defs=state.optimal_params_cols,
                        height='225px',
                        width='900px',
                        theme='balham',
                    ),
                    rx.divider(width='900px'),
                ),
                position='fixed',
                left='400px',
                top='90px',
            ),
            rx.vstack(
                rx.heading('Value bets'),
                rx.vstack(
                    rx.divider(width='900px'),
                    ag_grid(
                        id='value_bets',
                        row_data=state.value_bets,
                        column_defs=state.value_bets_cols,
                        height='460px',
                        width='900px',
                        theme='balham',
                    ),
                    rx.divider(width='900px'),
                ),
                position='fixed',
                left='400px',
                top='90px',
            ),
        ),
    )


def bot(state: rx.State, visibility_level: int) -> rx.Component:
    """The bot component."""
    return rx.cond(
        state.visibility_level < visibility_level,
        rx.box(
            rx.vstack(
                rx.icon('bot-message-square', size=70),
                rx.html(state.streamed_message),
            ),
            position='fixed',
            left='600px',
            top='100px',
            width='500px',
            background_color=rx.color('gray', 2),
            padding='30px',
        ),
    )
