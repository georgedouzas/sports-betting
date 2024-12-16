"""Page common components."""

from collections.abc import Callable

import reflex as rx


def home(reset_state: Callable) -> rx.Component:
    """Home title."""
    return rx.link(rx.hstack(rx.icon('home'), rx.text('Home', size='4', weight='bold')), on_click=reset_state)


def header() -> rx.Component:
    """Header of page."""
    return rx.vstack(
        rx.heading("Sports Betting", size='9', align='center'),
        rx.heading("Application", size='4', color_scheme='blue'),
        spacing='1',
        align='center',
    )


def title(name: str, icon_name: str) -> rx.Component:
    """The title component."""
    return rx.hstack(
        rx.icon(icon_name),
        rx.text(name, size="4"),
        width="100%",
        padding_y="0.75rem",
        color=rx.color("accent", 11),
    )


def selection(items: list[str], value: str, disabled: bool, on_change: Callable) -> rx.Component:
    """The selection component."""
    return rx.select(
        items=items,
        value=value,
        disabled=disabled,
        on_change=on_change,
        width='50%',
    )
