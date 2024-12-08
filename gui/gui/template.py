"""Implementation of template."""

from collections.abc import Callable

import reflex as rx

from .components.header import header

SIDEBAR_OPTIONS = {
    'spacing': '5',
    'position': 'fixed',
    'left': '50px',
    'top': '200px',
    'padding_x': '1em',
    'padding_y': "1.5em",
    'bg': rx.color('blue', 3),
    'height': '650px',
    'width': '35em',
}


def template(main_name: str, main_icon: str, secondary_name: str, secondary_icon: str) -> Callable:
    """Template of pages."""

    def _template(
        page: tuple[Callable[[], rx.Component]],
    ) -> rx.Component:
        main_component, control_component = page()
        return rx.box(
            header(),
            rx.box(
                rx.vstack(
                    rx.hstack(rx.icon(main_icon), rx.text(main_name, size='4', weight='bold')),
                    rx.divider(),
                    rx.hstack(
                        rx.icon(secondary_icon),
                        rx.text(secondary_name, size="4"),
                        width="100%",
                        padding_y="0.75rem",
                        color=rx.color("accent", 11),
                    ),
                    main_component,
                    control_component,
                    **SIDEBAR_OPTIONS,
                ),
            ),
        )

    return _template
