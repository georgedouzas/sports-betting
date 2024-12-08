"""Index page."""

import reflex as rx

from ..template import template


class ToolboxState(rx.State):
    """The toolbox state."""

    tool: str | None = None


def main_component() -> rx.Component:
    """The main component."""
    return rx.select(
        ['Data', 'Modelling'],
        width='50%',
        placeholder='Selection of tool',
        on_change=ToolboxState.set_tool,
    )


def control_component() -> rx.Component:
    """The control component."""
    return rx.cond(
        ToolboxState.tool,
        rx.cond(
            ToolboxState.tool == 'Data',
            rx.link(rx.icon_button('check'), href='dataloader-selection'),
            rx.link(rx.icon_button('check'), href='model-selection'),
        ),
        rx.tooltip(rx.icon_button('check', disabled=True), content='Please select a tool'),
    )


@rx.page(route="/")
@template('Home', 'home', 'Toolbox', 'hammer')
def index() -> rx.Component:
    """Main page."""
    return main_component(), control_component()
