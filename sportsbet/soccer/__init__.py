"""
Defines the soccer data path.
"""

from os.path import join
from pathlib import Path

from .. import PATH

SOCCER_PATH = join(PATH, 'soccer')
Path(SOCCER_PATH).mkdir(exist_ok=True)


