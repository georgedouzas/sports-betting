"""Equivalence gate against the reference fingerprint.

Marked `network` and deselected by default, since it reads the live feed. It is the check that the data source can be
replaced without changing a single extracted value.
"""

import pytest

from tests.reference import extract_frames, fingerprint, load_fingerprint

REFERENCE = load_fingerprint()

pytestmark = pytest.mark.network


@pytest.fixture(scope='module')
def frames():
    """Extract the reference frames from the current data source."""
    return extract_frames()


@pytest.mark.parametrize('name', REFERENCE['frames'])
def test_frame_matches_reference(frames, name):
    """Test the extracted frame matches the reference columns, dtypes, shape and values."""
    reference = REFERENCE['frames'][name]
    current = fingerprint(frames[name])
    assert current['columns'] == reference['columns']
    assert current['dtypes'] == reference['dtypes']
    assert current['shape'] == reference['shape']
    drifted = [col for col, digest in reference['hashes'].items() if current['hashes'][col] != digest]
    assert not drifted, f'The {name} columns drifted from the reference: {drifted}.'
