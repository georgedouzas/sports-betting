"""Test the main commands."""

from sportsbet.cli import main


def test_main(cli_runner):
    """Test main command."""
    result = cli_runner.invoke(main, [])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'CLI for sports-betting' in result.output


def test_main_help(cli_runner):
    """Test main help command."""
    result = cli_runner.invoke(main, ['--help'])
    assert result.exit_code == 0, result.output
    assert 'CLI for sports-betting' in result.output
