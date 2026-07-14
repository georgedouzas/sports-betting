"""Reference capture and fingerprinting for the equivalence gate.

Run with `python -m tests.reference` against the live feed to refresh the reference. It fingerprints the extracted
frames as column names, dtypes, shapes and per-column hashes, and writes them to `samples/reference_fingerprint.json`.

Only the fingerprint is committed. The frames it is computed from, and the raw upstream inputs, carry third-party odds
and stay out of the repository.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import FootballDataOdds, FootballDataStats, LocalStore

PARAM_GRID = {'league': ['England'], 'division': [1], 'year': [2025]}
RAW_URLS = {
    'E0_2425.csv': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
    'fixtures.csv': 'https://www.football-data.co.uk/fixtures.csv',
}
SAMPLES_PATH = Path(__file__).parent / 'samples'
FINGERPRINT_PATH = SAMPLES_PATH / 'reference_fingerprint.json'
RAW_PATH = SAMPLES_PATH / 'football_data'
FRAMES_PATH = SAMPLES_PATH / 'reference_frames'
STORE_PATH = SAMPLES_PATH / 'store'


def hash_column(column: pd.Series) -> str:
    """Return a stable hash of a column's values."""
    content = column.to_csv(index=False, float_format='%.10g')
    return hashlib.sha256(content.encode()).hexdigest()


def fingerprint(data: pd.DataFrame) -> dict[str, Any]:
    """Reduce a frame to its column names, dtypes, shape and per-column hashes."""
    data = data.reset_index(drop=data.index.name is None)
    return {
        'columns': data.columns.tolist(),
        'dtypes': [str(dtype) for dtype in data.dtypes],
        'shape': list(data.shape),
        'hashes': {col: hash_column(data[col]) for col in data.columns},
    }


def extract_frames() -> dict[str, pd.DataFrame]:
    """Extract the reference frames from the current data source."""
    loader = DataLoader(
        param_grid=PARAM_GRID,
        stats=FootballDataStats(),
        odds=FootballDataOdds(),
        store=LocalStore(STORE_PATH),
    )
    loader.prepare()
    X, Y, O_average = loader.extract_train_data(odds_type='market_average')
    _, _, O_maximum = loader.extract_train_data(odds_type='market_maximum')
    return {'stats': loader.stats_, 'X': X, 'Y': Y, 'O_market_average': O_average, 'O_market_maximum': O_maximum}


def load_fingerprint() -> dict[str, Any]:
    """Load the committed reference fingerprint."""
    with FINGERPRINT_PATH.open() as fingerprint_file:
        reference: dict[str, Any] = json.load(fingerprint_file)
    return reference


async def _download(urls: list[str]) -> list[str]:
    """Download the URLs content."""

    async with aiohttp.ClientSession(raise_for_status=True) as client:
        return [await (await client.get(url)).text(encoding='ISO-8859-1') for url in urls]


def record_raw() -> None:
    """Record the raw upstream inputs the client-side sources are checked against."""
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    contents = asyncio.run(_download(list(RAW_URLS.values())))
    for name, content in zip(RAW_URLS, contents, strict=True):
        with (RAW_PATH / name).open('w', encoding='ISO-8859-1') as raw_file:
            raw_file.write(content)


def main() -> None:
    """Capture the reference frames and write their fingerprint."""
    record_raw()
    frames = extract_frames()
    FRAMES_PATH.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(FRAMES_PATH / f'{name}.parquet')
    reference = {'param_grid': PARAM_GRID, 'frames': {name: fingerprint(frame) for name, frame in frames.items()}}
    with FINGERPRINT_PATH.open('w') as fingerprint_file:
        json.dump(reference, fingerprint_file, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
