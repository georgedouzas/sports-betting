"""Run one cell of an example page and render what it shows."""

import ast
import builtins
import contextlib
import functools
import html
import io
from typing import Any

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt


def _render_value(value: object) -> str | None:
    """Render the value of a cell's last expression, as a notebook would."""
    if value is None or type(value).__module__.startswith('matplotlib'):
        return None
    to_html = getattr(value, '_repr_html_', None)
    if callable(to_html):
        rendered = to_html()
        if isinstance(rendered, str):
            return rendered
    return f'<pre>{html.escape(repr(value))}</pre>'


def _render_figures() -> list[str]:
    """Render each open figure as inline SVG, then close them all."""
    figures = []
    for number in plt.get_fignums():
        buffer = io.StringIO()
        plt.figure(number).savefig(buffer, format='svg', bbox_inches='tight')
        figures.append(buffer.getvalue())
    plt.close('all')
    return figures


def run(source: str, namespace: dict[str, Any]) -> str:
    """Run a cell in the page's namespace and return what it shows as HTML.

    The printed output, the value of the last expression and the open figures are shown in that order. markdown-exec
    puts its own `print` in the namespace, so the cell's `print` is pointed at the buffer while the cell runs. An error
    is not caught, so markdown-exec reports it and the strict build fails.

    Args:
        source:
            The code of the cell.
        namespace:
            The globals the page's cells share.

    Returns:
        The HTML to show below the cell.
    """
    tree = ast.parse(source)
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        tree.body.pop()
    else:
        last = None
    printed = io.StringIO()
    page_print = namespace.get('print')
    namespace['print'] = functools.partial(builtins.print, file=printed)
    value = None
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(tree, '<example>', 'exec'), namespace)  # noqa: S102  # nosec B102
            if last is not None:
                expression = compile(ast.Expression(last.value), '<example>', 'eval')
                value = eval(expression, namespace)  # noqa: S307  # nosec B307
    finally:
        namespace['print'] = page_print
    parts = []
    if printed.getvalue():
        parts.append(f'<pre>{html.escape(printed.getvalue())}</pre>')
    rendered = _render_value(value)
    if rendered is not None:
        parts.append(rendered)
    parts.extend(_render_figures())
    return '\n'.join(parts)
