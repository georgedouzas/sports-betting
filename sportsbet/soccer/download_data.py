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


def generate_spi_dataset():
    """Download and transform the dataset containing match-by-match SPI ratings."""
    spi_data = pd.read_csv(SPI_URL)
    spi_data = spi_data.loc[(~spi_data.score1.isna()) & (spi_data.league.isin(LEAGUES_MAPPING.keys())), SPI_DATA_FEATURES].reset_index(drop=True)
    spi_data['league'] = spi_data['league'].apply(lambda league: LEAGUES_MAPPING[league])
    spi_data['date'] = pd.to_datetime(spi_data['date'], format='%Y-%m-%d')
    spi_data.rename(columns=SPI_FEATURES_MAPPING, inplace=True)
    return spi_data


def download_fd_dataset():
    """Download and transform the datasets from football-data.co.uk."""
    fd_dataset = pd.DataFrame()
    for url in FD_URLS:
        data = pd.read_csv(url)
        season = sub('^' + FD_URL, '', url).split('/')[1]
        data['Season'] = season[:2] + '-' + season[2:]
        fd_dataset = fd_dataset.append(data.loc[:, FD_DATA_FEATURES])
    fd_dataset.reset_index(drop=True, inplace=True)
    return fd_dataset


def generate_fd_dataset(fd_dataset):
    """Transform the datasets from football-data.co.uk."""
    probs = 1 / fd_dataset.loc[:, FD_AVG_ODDS_FEATURES].values
    probs = pd.DataFrame(probs / probs.sum(axis=1)[:, None], columns=PROB_FD_FEATURES)
    fd_data = pd.concat([probs, fd_dataset], axis=1)
    fd_data['Date'] = pd.to_datetime(fd_data['Date'], format='%d/%m/%y')
    fd_data['FTR'] = fd_data['FTR'].apply(lambda result: RESULTS_MAPPING[result])
    fd_data.drop(columns=FD_AVG_ODDS_FEATURES, inplace=True)
    fd_data.rename(columns=FD_FEATURES_MAPPING, inplace=True)
    return fd_data


def generate_odds_dataset(fd_dataset):
    """Extract the odds data from football-data.co.uk"""
    fd_dataset = fd_dataset.rename(columns={'Div': 'League'})
    odds_data = fd_dataset.loc[:, ['Season'] + KEYS_FEATURES + FD_MAX_ODDS_FEATURES]
    odds_data['Date'] = pd.to_datetime(odds_data['Date'], format='%d/%m/%y')
    odds_data = odds_data.sort_values(KEYS_FEATURES).drop(columns=['Date'])
    return odds_data


def generate_training_dataset(spi_data, fd_data):
    """Combine datasets to generate a training dataset."""
    fd_teams = pd.unique(fd_data.HomeTeam)
    teams_mapping = {}
    for team in pd.unique(spi_data.HomeTeam):
        closer_match = get_close_matches(team, fd_teams)
        if closer_match:
            teams_mapping[team] = closer_match[0]
    teams_mapping.update(TEAMS_MAPPING)
    spi_data['HomeTeam'] = spi_data['HomeTeam'].apply(lambda team: teams_mapping[team])
    spi_data['AwayTeam'] = spi_data['AwayTeam'].apply(lambda team: teams_mapping[team])
    training = pd.merge(spi_data, fd_data, on=KEYS_FEATURES)
    training.sort_values(KEYS_FEATURES, inplace=True)
    training['Day'] = (training.Date - min(training.Date)).dt.days
    training = training[TRAINING_FEATURES]
    return training


if __name__ =='__main__':
    spi_data = generate_spi_dataset()
    fd_dataset = download_fd_dataset()
    fd_data, odds_data = generate_fd_dataset(fd_dataset), generate_odds_dataset(fd_dataset)
    training = generate_training_dataset(spi_data, fd_data)
    training.to_csv(join(dirname(__file__), '..', '..', 'data', 'training.csv'), index=False)
    odds_data.to_csv(join(dirname(__file__), '..', '..', 'data', 'odds_data.csv'), index=False)
