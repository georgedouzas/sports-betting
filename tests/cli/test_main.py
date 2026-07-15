"""Test the main command group."""

from sportsbet.cli import main


def test_main(cli_runner):
    """Test the group prints its help when no command is given."""
    result = cli_runner.invoke(main, [])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'Create, test and use sports betting models.' in result.output


def test_main_help(cli_runner):
    """Test the group help lists the two command groups that mirror the packages."""
    result = cli_runner.invoke(main, ['--help'])
    assert result.exit_code == 0, result.output
    assert 'Create, test and use sports betting models.' in result.output
    for group in ('dataloader', 'evaluation'):
        assert group in result.output
