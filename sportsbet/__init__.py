"""
Includes definitions of the data paths.
"""

from os.path import join, dirname
from pathlib import Path
from sys import path

# Define paths
PATH = join(dirname(__file__), '..', 'data')
SOCCER_PATH = join(PATH, 'soccer')

# Create paths
Path(PATH).mkdir(exist_ok=True)
Path(SOCCER_PATH).mkdir(exist_ok=True)






