"""
Defines the data path.
"""

from pathlib import Path
from os.path import join
from sys import path

PATH = join(str(Path.home()), '.sports-betting')
path.append(PATH)





