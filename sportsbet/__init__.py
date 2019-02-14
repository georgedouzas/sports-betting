"""
Defines the data path.
"""

from os.path import join, dirname
from pathlib import Path

PATH = join(dirname(__file__), '..', 'data')
Path(PATH).mkdir(exist_ok=True)






