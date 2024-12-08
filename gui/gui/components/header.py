"""Header components."""

import reflex as rx


def header() -> rx.Component:
    """Header of page."""
    return rx.vstack(
        rx.heading("Sports Betting", size='9', align='center'),
        rx.heading("Application", size='4', color_scheme='blue'),
        spacing='1',
        align='center',
    )
