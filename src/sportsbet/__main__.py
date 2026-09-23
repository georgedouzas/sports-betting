"""Entry-point module."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys

from sportsbet.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
