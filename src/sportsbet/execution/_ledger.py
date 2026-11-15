"""Record each bet before and after it is placed, so a rerun never places it twice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import pandera.pandas as pa

from ._base import PlacementIntent, PlacementReceipt, PlacementStatus

_logger = logging.getLogger('sportsbet.execution')

ATTEMPTED = 'attempted'
PLACED_STATUSES = frozenset(
    {PlacementStatus.ACCEPTED.value, PlacementStatus.MATCHED_FULL.value, PlacementStatus.MATCHED_PARTIAL.value},
)
UNSETTLED_STATUSES = frozenset({ATTEMPTED, PlacementStatus.REFUSED_UNCONFIRMED.value})


class _LedgerSchema(pa.DataFrameModel):
    """The lines a ledger holds."""

    ref: str = pa.Field()
    status: str = pa.Field()
    written_at: Annotated[pd.DatetimeTZDtype, 'ns', 'utc'] = pa.Field()
    match: str = pa.Field()
    market: str = pa.Field()
    selection: str = pa.Field()
    stake: float = pa.Field(ge=0.0)
    price: float = pa.Field(nullable=True)
    venue_bet_id: str = pa.Field(nullable=True)

    class Config:
        """Allow no column beyond the ones named, and coerce the dtypes."""

        strict = True
        coerce = True


class _Ledger:
    """An append-only file of JSON lines, one per bet attempted or answered."""

    def __init__(self: _Ledger, path: str | Path) -> None:
        self.path = Path(path)

    def _append(self: _Ledger, line: dict) -> None:
        """Write one line and make sure it reached the disk before returning."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as ledger:
            ledger.write(json.dumps(line) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())

    def record_attempt(self: _Ledger, intent: PlacementIntent, now: pd.Timestamp) -> None:
        """Record that a bet is about to be placed."""
        identity = intent.identity
        self._append(
            {
                'ref': identity.ref_,
                'status': ATTEMPTED,
                'written_at': now.isoformat(),
                'match': identity.match,
                'market': identity.market,
                'selection': identity.selection,
                'stake': intent.stake,
                'price': None,
                'venue_bet_id': None,
            },
        )

    def record_receipt(self: _Ledger, receipt: PlacementReceipt, now: pd.Timestamp) -> None:
        """Record what became of a bet."""
        identity = receipt.identity
        self._append(
            {
                'ref': identity.ref_,
                'status': receipt.status.value,
                'written_at': now.isoformat(),
                'match': identity.match,
                'market': identity.market,
                'selection': identity.selection,
                'stake': receipt.stake,
                'price': receipt.price,
                'venue_bet_id': receipt.venue_bet_id,
            },
        )

    def read(self: _Ledger) -> pd.DataFrame:
        """Return every line the ledger holds, skipping a last line a crash left torn."""
        records = []
        if self.path.exists():
            lines = self.path.read_text(encoding='utf-8').splitlines()
            for number, line in enumerate(lines, start=1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if number < len(lines):
                        raise
                    _logger.info('The last line of %s is torn, so it was skipped.', self.path)
        frame = pd.DataFrame.from_records(records, columns=list(_LedgerSchema.to_schema().columns))
        frame['written_at'] = pd.to_datetime(frame['written_at'], utc=True)
        result: pd.DataFrame = _LedgerSchema.validate(frame)
        return result

    def read_status(self: _Ledger, ref: str) -> str | None:
        """Return the latest status the ledger holds for a bet, or `None` when it holds none."""
        lines = self.read()
        held = lines.loc[lines['ref'] == ref, 'status']
        return None if held.empty else str(held.iloc[-1])
