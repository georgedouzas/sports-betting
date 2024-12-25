"""Page common components."""

import reflex as rx

SIDEBAR_OPTIONS = {
    'spacing': '1',
    'position': 'fixed',
    'left': '50px',
    'top': '50px',
    'padding_x': '1em',
    'padding_y': "1.5em",
    'bg': rx.color('blue', 3),
    'height': '620px',
    'width': '20em',
}


def home() -> rx.Component:
    """Home title."""
    return rx.text('Sports Betting', size='4', weight='bold')


def title(name: str, icon_name: str) -> rx.Component:
    """The title component."""
    return rx.hstack(
        rx.icon(icon_name),
        rx.text(name, size="4"),
        width="100%",
        padding_y="0.75rem",
        color=rx.color("accent", 11),
    )


def select_mode(state: rx.State, content: str) -> rx.Component:
    """Selection of mode component."""
    return rx.vstack(
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


def control_buttons(state: rx.State, disabled: bool) -> rx.Component:
    """Control buttons of UI."""
    return rx.vstack(
        rx.divider(top='600px', position='fixed', width='18em'),
        rx.hstack(
            rx.button(
                'Submit',
                on_click=state.submit_state,
                disabled=disabled,
                loading=state.loading,
                position='fixed',
                top='620px',
                width='70px',
            ),
            rx.link(
                rx.button(
                    'Reset',
                    on_click=state.reset_state,
                    position='fixed',
                    top='620px',
                    left='150px',
                    width='70px',
                ),
                href='/',
            ),
        ),
    )
