"""Index page."""

from typing import Self

import reflex as rx

from .components import SIDEBAR_OPTIONS, home, title


class State(rx.State):
    """The toolbox state."""

    # Elements
    visibility_level: int = 1
    loading: bool = False

    # Mode
    mode_category: str = 'Data'
    mode_type: str = 'Create'

    def submit_state(self: Self) -> None:
        """Submit handler."""
        self.loading = True
        yield
        if self.visibility_level == 1:
            self.loading = False
            yield
        self.visibility_level += 1

    def reset_state(self: Self) -> None:
        """Reset handler."""

        # Elements visibility
        self.visibility_level = 1
        self.loading: bool = False

        # Mode
        self.mode_category = 'Data'
        self.mode_type = 'Create'


@rx.page(route="/")
def index() -> rx.Component:
    """Main page."""
    return rx.container(
        rx.vstack(
            home(),
            rx.divider(),
            # Mode selection
            title('Mode', 'blend'),
            rx.vstack(
                rx.cond(
                    (State.mode_category == 'Data') & (State.mode_type == 'Create'),
                    rx.text('Create a dataloader', size='1'),
                ),
                rx.cond(
                    (State.mode_category == 'Data') & (State.mode_type == 'Load'),
                    rx.text('Load a dataloader', size='1'),
                ),
                rx.cond(
                    (State.mode_category == 'Modelling') & (State.mode_type == 'Create'),
                    rx.text('Create a model', size='1'),
                ),
                rx.cond(
                    (State.mode_category == 'Modelling') & (State.mode_type == 'Load'),
                    rx.text('Load a model', size='1'),
                ),
                rx.hstack(
                    rx.select(
                        items=['Data', 'Modelling'],
                        value=State.mode_category,
                        disabled=State.visibility_level > 1,
                        width='120px',
                        on_change=State.set_mode_category,
                    ),
                    rx.select(
                        ['Create', 'Load'],
                        value=State.mode_type,
                        disabled=State.visibility_level > 1,
                        width='120px',
                        on_change=State.set_mode_type,
                    ),
                ),
            ),
            # Control
            rx.vstack(
                rx.divider(top='600px', position='fixed', width='18em'),
                rx.hstack(
                    rx.cond(
                        (State.mode_category == 'Data') & (State.mode_type == 'Create'),
                        rx.link(
                            rx.button(
                                'Submit',
                                on_click=State.submit_state,
                                loading=State.loading,
                                position='fixed',
                                top='620px',
                                width='70px',
                            ),
                            href='/dataloader/creation',
                        ),
                    ),
                    rx.cond(
                        (State.mode_category == 'Data') & (State.mode_type == 'Load'),
                        rx.link(
                            rx.button(
                                'Submit',
                                on_click=State.submit_state,
                                loading=State.loading,
                                position='fixed',
                                top='620px',
                                width='70px',
                            ),
                            href='/dataloader/loading',
                        ),
                    ),
                    rx.cond(
                        (State.mode_category == 'Modelling') & (State.mode_type == 'Create'),
                        rx.link(
                            rx.button(
                                'Submit',
                                on_click=State.submit_state,
                                loading=State.loading,
                                position='fixed',
                                top='620px',
                                width='70px',
                            ),
                            href='/model/creation',
                        ),
                    ),
                    rx.cond(
                        (State.mode_category == 'Modelling') & (State.mode_type == 'Load'),
                        rx.link(
                            rx.button(
                                'Submit',
                                on_click=State.submit_state,
                                loading=State.loading,
                                position='fixed',
                                top='620px',
                                width='70px',
                            ),
                            href='/model/loading',
                        ),
                    ),
                    rx.link(
                        rx.button(
                            'Reset',
                            on_click=State.reset_state,
                            position='fixed',
                            top='620px',
                            left='150px',
                            width='70px',
                        ),
                        href='/',
                    ),
                ),
            ),
            **SIDEBAR_OPTIONS,
        ),
    )
