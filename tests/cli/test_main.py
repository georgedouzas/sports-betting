"""Test the main commands."""

import pytest

from sportsbet.cli import main


def test_main(capsys):
    """Test main command."""
    with pytest.raises(SystemExit):
        main([])
    captured = capsys.readouterr()
    assert 'CLI for sports-betting' in captured.out


def test_main_help(capsys):
    """Test main help command."""
    with pytest.raises(SystemExit):
        main(['--help'])
    captured = capsys.readouterr()
    assert 'CLI for sports-betting' in captured.out
