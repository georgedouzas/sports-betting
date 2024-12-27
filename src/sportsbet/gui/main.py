"""Script to run the GUI application."""

import subprocess
import sys
from pathlib import Path

import click


def run() -> None:
    """Run the GUI application."""
    null = subprocess.DEVNULL
    node_call = subprocess.run(['node', '--version'], stdout=null, stderr=null, check=False)  # noqa: S603, S607
    if node_call.returncode == 0:
        reflex_path = Path(sys.executable).parent / 'reflex'
        subprocess.run([reflex_path, 'run'], cwd=f'{Path(__file__).parent}', check=True)  # noqa: S603
    else:
        click.echo('Node executable not found. Please install it to proceed.')


if __name__ == '__main__':
    run()
