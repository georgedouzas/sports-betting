"""Extract sports betting data, build predictive models, and place the bets they choose.

The library has four submodules:

- [`sources`][sportsbet.sources]: Read match statistics and odds from data feeds.
- [`dataloaders`][sportsbet.dataloaders]: Shape the source data into modelling data.
- [`evaluation`][sportsbet.evaluation]: Build and evaluate betting models.
- [`execution`][sportsbet.execution]: Place the chosen bets at a venue.

The same capabilities are reached from the `sportsbet` command line and from the `sportsbet-mcp` server.
"""
