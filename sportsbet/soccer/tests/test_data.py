"""
Test the data module.
"""

import requests

from sportsbet.soccer.data import SPIDataSource, FDDataSource

SPI_DATA_SOURCE = SPIDataSource(['E0', 'G1']).download()


def test_spi_connection():
    """Test SPI data source connection."""
    status_code = requests.head(SPIDataSource.url).status_code
    assert status_code == 200


def test_spi_initialization():
    """Test SPI initialization."""
    assert SPI_DATA_SOURCE.url == 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    assert SPI_DATA_SOURCE.match_cols == ['date', 'league', 'team1', 'team2']
    assert SPI_DATA_SOURCE.full_goals_cols == ['score1', 'score2']
    assert SPI_DATA_SOURCE.spi_cols == ['spi1', 'spi2', 'prob1', 'probtie', 'prob2', 'proj_score1', 'proj_score2']
    assert SPI_DATA_SOURCE.leagues_ids == ['E0', 'G1']


def test_spi_download():
    """Test SPI download."""
    assert hasattr(SPI_DATA_SOURCE, 'content_')
    assert set(SPI_DATA_SOURCE.content_.columns) == set(SPI_DATA_SOURCE.match_cols + SPI_DATA_SOURCE.full_goals_cols + SPI_DATA_SOURCE.spi_cols)


def test_spi_transform():
    """Test SPI transform."""
    content = SPI_DATA_SOURCE.content_
    print(type(content.date))


def test_fd_connection():
    """Test FD data sources connection."""
    status_code = requests.head(FDDataSource.base_url).status_code
    assert status_code == 200

    
