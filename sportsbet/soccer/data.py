#!/usr/bin/env python

"""
Download and prepare soccer historical data from various leagues.
"""

from os.path import join, dirname
from re import sub
from difflib import get_close_matches
import pandas as pd
from sportsbet.soccer import (
    LEAGUES_MAPPING,
    SPI_FEATURES_MAPPING,
    SPI_URL,
    SPI_DATA_FEATURES,
    FD_URL,
    FD_URLS,
    FD_DATA_FEATURES,
    FD_AVG_ODDS_FEATURES,
    FD_MAX_ODDS_FEATURES,
    PROB_FD_FEATURES,
    FD_FEATURES_MAPPING,
    TEAMS_MAPPING,
    TRAINING_FEATURES,
    KEYS_FEATURES,
    RESULTS_MAPPING,
)


def fetch_raw_data():
    """Fetch the data containing match-by-match SPI ratings
    and the data from football-data.co.uk."""

    # SPI data
    spi_data = pd.read_csv(SPI_URL)
    spi_data = spi_data.loc[(~spi_data.score1.isna()) & (spi_data.league.isin(LEAGUES_MAPPING.keys())), SPI_DATA_FEATURES].reset_index(drop=True)
    spi_data['league'] = spi_data['league'].apply(lambda league: LEAGUES_MAPPING[league])
    spi_data['date'] = pd.to_datetime(spi_data['date'], format='%Y-%m-%d')
    spi_data.rename(columns=SPI_FEATURES_MAPPING, inplace=True)

    # Football data
    fd_data = pd.DataFrame()
    for url in FD_URLS:
        data = pd.read_csv(url)
        season = sub('^' + FD_URL, '', url).split('/')[1]
        data['Season'] = season[:2] + '-' + season[2:]
        fd_data = fd_data.append(data.loc[:, FD_DATA_FEATURES])
    fd_data.reset_index(drop=True, inplace=True)

    return spi_data, fd_data


def extract_training_data(spi_data, fd_data):
    """Extract the training dataset."""

    # Probabilities data
    probs = 1 / fd_data.loc[:, FD_AVG_ODDS_FEATURES].values
    probs = pd.DataFrame(probs / probs.sum(axis=1)[:, None], columns=PROB_FD_FEATURES)
    probs_data = pd.concat([probs, fd_data], axis=1)
    probs_data['Date'] = pd.to_datetime(probs_data['Date'], format='%d/%m/%y')
    probs_data['FTR'] = probs_data['FTR'].apply(lambda result: RESULTS_MAPPING[result])
    probs_data.drop(columns=FD_AVG_ODDS_FEATURES, inplace=True)
    probs_data.rename(columns=FD_FEATURES_MAPPING, inplace=True)

    # SPI data
    fd_teams = pd.unique(probs_data.HomeTeam)
    teams_mapping = {}
    for team in pd.unique(spi_data.HomeTeam):
        closer_match = get_close_matches(team, fd_teams)
        if closer_match:
            teams_mapping[team] = closer_match[0]
    teams_mapping.update(TEAMS_MAPPING)
    spi_data['HomeTeam'] = spi_data['HomeTeam'].apply(lambda team: teams_mapping.get(team))
    spi_data['AwayTeam'] = spi_data['AwayTeam'].apply(lambda team: teams_mapping.get(team))

    # Combine data
    training_data = pd.merge(spi_data, probs_data, on=KEYS_FEATURES)
    training_data.sort_values(KEYS_FEATURES, inplace=True)
    training_data['Day'] = (training_data.Date - min(training_data.Date)).dt.days
    training_data = training_data[TRAINING_FEATURES]

    return training_data


def extract_odds_dataset(fd_data):
    """Extract the odds data from"""
    fd_data = fd_data.rename(columns={'Div': 'League'})
    odds_data = fd_data.loc[:, ['Season'] + KEYS_FEATURES + FD_MAX_ODDS_FEATURES]
    odds_data['Date'] = pd.to_datetime(odds_data['Date'], format='%d/%m/%y')
    odds_data = odds_data.sort_values(KEYS_FEATURES).drop(columns=['Date'])
    return odds_data


if __name__ == '__main__':
    spi_data, fd_data = fetch_raw_data()
    training_data = extract_training_data(spi_data, fd_data)
    odds_data = extract_odds_dataset(fd_data)
    training_data.to_csv(join(dirname(__file__), '..', '..', 'data', 'training_data.csv'), index=False)
    odds_data.to_csv(join(dirname(__file__), '..', '..', 'data', 'odds_data.csv'), index=False)
