"""Test the single-event execution command drives the runner offline."""

from sportsbet.cli import main
from sportsbet.execution import BetIdentity, PlacementReceipt, PlacementStatus, build_receipts_frame

STAKE = 10.0


def test_run_dry_run_prints_the_bet_it_would_make(cli_runner, monkeypatch, tmp_path):
    """A dry run loads the inputs, runs the unit unarmed, and prints the receipt it would place."""
    dataloader = tmp_path / 'loader.pkl'
    dataloader.write_text('loader')
    bettor = tmp_path / 'model.pkl'
    bettor.write_text('model')
    calls: dict[str, object] = {}

    async def fake_execute_event(event, bettor_arg, loader_arg, session, *, stake, urls, live, poll):
        calls.update(event=event, stake=stake, urls=urls, live=live)
        receipt = PlacementReceipt(
            identity=BetIdentity('stub', event, 'home_win', 'Arsenal'),
            status=PlacementStatus.DRY_RUN,
            value_bet=f'{event}|home_win',
            detail='Dry run, so nothing was staked.',
        )
        return build_receipts_frame([receipt])

    monkeypatch.setattr('sportsbet.cli._execution._load_session', lambda ref: object())
    monkeypatch.setattr('sportsbet.cli._execution.load_dataloader', lambda path: object())
    monkeypatch.setattr('sportsbet.cli._execution.load_bettor', lambda path: object())
    monkeypatch.setattr('sportsbet.cli._execution.execute_event', fake_execute_event)

    result = cli_runner.invoke(
        main,
        [
            'execution',
            'run',
            '--venue',
            'venue.py:VENUE',
            '-d',
            str(dataloader),
            '-b',
            str(bettor),
            '--event',
            'Arsenal vs Chelsea',
            '--stake',
            str(STAKE),
            '--url',
            'https://book.invalid/ac',
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls['live'] is False
    assert calls['event'] == 'Arsenal vs Chelsea'
    assert calls['stake'] == STAKE
    assert 'Arsenal' in result.output
