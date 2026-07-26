"""Provide the types, constants and building primitives the whole library builds on."""

from ._errors import BuildError
from ._params import (
    ALIASES,
    DATE_COLS,
    EVENT_COLS,
    GROUPS_COLS,
    IDENTITY_COLS,
    IDENTITY_FIELDS,
    INPUT_EVENT_STATUSES,
    MATCH_COLS,
    STATUSES,
    TARGET_EVENT_STATUSES,
    TEAMS_COLS,
)
from ._types import (
    BoolData,
    Data,
    FixturesData,
    Indices,
    OutputsMapping,
    Param,
    ParamGrid,
    Schema,
    TrainData,
)
from ._utils import load_object

__all__ = [
    'ALIASES',
    'DATE_COLS',
    'EVENT_COLS',
    'GROUPS_COLS',
    'IDENTITY_COLS',
    'IDENTITY_FIELDS',
    'INPUT_EVENT_STATUSES',
    'MATCH_COLS',
    'STATUSES',
    'TARGET_EVENT_STATUSES',
    'TEAMS_COLS',
    'BoolData',
    'BuildError',
    'Data',
    'FixturesData',
    'Indices',
    'OutputsMapping',
    'Param',
    'ParamGrid',
    'Schema',
    'TrainData',
    'load_object',
]
