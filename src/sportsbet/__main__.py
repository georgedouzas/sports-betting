"""Entry-point module."""

import sys

from sportsbet.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
