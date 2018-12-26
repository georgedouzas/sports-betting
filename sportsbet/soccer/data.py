"""
Download and prepare training and fixtures data 
from various leagues.
"""

from os.path import join
from itertools import product
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base import BaseDataSource, BaseDataLoader

LEAGUES_MAPPING = {
    'E0': 'Barclays Premier League',
    'B1': 'Belgian Jupiler League',
    'N1': 'Dutch Eredivisie',
    'E1': 'English League Championship',
    'E2': 'English League One',
    'E3': 'English League Two',
    'F1': 'French Ligue 1',
    'F2': 'French Ligue 2',
    'D1': 'German Bundesliga',
    'D2': 'German 2. Bundesliga',
    'G1': 'Greek Super League',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'P1': 'Portuguese Liga',
    'SC0': 'Scottish Premiership',
    'SP1': 'Spanish Primera Division',
    'SP2': 'Spanish Segunda Division',
    'T1': 'Turkish Turkcell Super Lig'
}


class SPIDataSource(BaseDataSource):

    url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    match_cols = ['date', 'league', 'team1', 'team2']
    full_goals_cols = ['score1', 'score2']
    spi_cols = ['spi1', 'spi2', 'prob1', 'probtie', 'prob2', 'proj_score1', 'proj_score2']

    def __init__(self, leagues_ids):
        self.leagues_ids = leagues_ids
    
    def download(self):
        """Download the data source."""
        columns = self.match_cols + self.full_goals_cols + self.spi_cols
        self.content_ = pd.read_csv(self.url, usecols=columns)

        return self

    def transform(self):
        """Transform the data source."""

        # Copy content
        content = self.content_.copy()

        # Cast to date
        content['date'] = pd.to_datetime(content['date'], format='%Y-%m-%d')

        # Filter leagues
        leagues = [LEAGUES_MAPPING[league_id] for league_id in self.leagues_ids if self.leagues_ids]
        content = content.loc[content['league'].isin(leagues)]

        # Convert league names to ids
        inverse_leagues_mapping = {league: league_id for league_id, league in LEAGUES_MAPPING.items()}
        content.loc[:, 'league'] = content.loc[:, 'league'].apply(lambda league: inverse_leagues_mapping[league])

        # Filter matches
        mask = (~content['score1'].isna()) & (~content['score2'].isna())
        content = [content[mask], content[~mask]]

        return content


class FDDataSource(BaseDataSource):

    base_url = 'http://www.football-data.co.uk'
    match_cols = ['Date', 'Div', 'HomeTeam', 'AwayTeam']
    full_goals_cols = ['FTHG', 'FTAG']
    half_goals_cols = ['HTHG', 'HTAG']
    avg_max_odds_cols = ['BbAvH', 'BbAvD', 'BbAvA', 'BbMxH', 'BbMxD', 'BbMxA']
    odds_cols = ['PSH', 'PSD', 'PSA', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']
    

class FDTrainingDataSource(FDDataSource):

    suffix_url = 'mmz4281'

    def __init__(self, league_id, season):
        self.league_id = league_id
        self.season = season

    def download(self):
        """Download the data source."""

        # Download content
        columns = self.match_cols + self.full_goals_cols + self.half_goals_cols + self.avg_max_odds_cols + self.odds_cols
        self.content_ = pd.read_csv(join(self.base_url, self.suffix_url, self.season, self.league_id), usecols=columns)

        return self

    def transform(self):
        """Transform the data source."""

        # Copy content
        content = self.content_.copy()

        # Cast to date
        content['Date'] = pd.to_datetime(content['Date'], dayfirst=True)

        # Create season column
        content['Season'] = self.season
        
        return content


class FDFixturesDataSource(FDDataSource):

    suffix_url = 'fixtures.csv'

    def __init__(self, leagues_ids):
        self.leagues_ids = leagues_ids

    def download(self):
        """Download the data source."""
        columns = self.match_cols + self.avg_max_odds_cols + self.odds_cols
        self.content_ = pd.read_csv(join(self.base_url, self.suffix_url), usecols=columns)
        return self

    def transform(self):
        """Transform the data source."""

        # Copy content
        content = self.content_.copy()

        # Cast to date
        content['Date'] = pd.to_datetime(content['Date'], dayfirst=True)

        # Filter leagues
        content = content.loc[content['Div'].isin(self.leagues_ids)]

        return content


class SoccerDataLoader(BaseDataLoader):

    seasons = ['1617', '1718', '1819']
    match_cols = ['Season'] + FDDataSource.match_cols
    input_data_cols = FDDataSource.avg_max_odds_cols + SPIDataSource.spi_cols
    goals_cols = FDDataSource.full_goals_cols + FDDataSource.half_goals_cols
    odds_cols = FDDataSource.odds_cols

    def __init__(self, leagues_ids, betting_type):
        self.leagues_ids = leagues_ids
        self.betting_type = betting_type

    def _merge_data(self, spi_data):
        """Merge data to convert names."""
        for column in ['team1', 'team2']:
            spi_data = pd.merge(spi_data, self.matching_, how='left', left_on=column, right_on='spi').drop(columns=[column, 'spi']).rename(columns={'fd': column})
        return spi_data

    def _rename_teams(self, spi_data):
        """Rename teams to match fd names."""
        
        if hasattr(self, 'matching_'):

            # Merge data
            spi_data = self._merge_data(spi_data)
            
            return spi_data

        # Generate teams names combinations
        teams_names_combinations = pd.merge(self.spi_training_data_, self.fd_training_data_, left_on=['date', 'league'], right_on=['Date', 'Div'], how='outer').loc[:, ['team1', 'HomeTeam', 'team2', 'AwayTeam']].dropna().reset_index(drop=True)

        # Calculate similarity index
        similarity = teams_names_combinations.apply(lambda row: SequenceMatcher(None, row[0], row[1]).ratio() * SequenceMatcher(None, row[2], row[3]).ratio(), axis=1)

        # Append similarity index
        teams_names_similarity = pd.concat([teams_names_combinations, similarity], axis=1)

        # Filter correct matches
        indices = teams_names_similarity.groupby(['team1', 'team2'])[0].idxmax().values
        teams_names_matching = teams_names_combinations.take(indices)

        # Home teams matching
        matching1 = teams_names_matching.iloc[:, 0:2]
        
        # Away teams matching
        matching2 = teams_names_matching.iloc[:, 2:]
        
        # Combine matching
        columns = ['spi', 'fd']
        matching1.columns, matching2.columns = columns, columns
        matching = matching1.append(matching2)
        matching = matching.groupby(columns).size().reset_index()
        indices = matching.groupby(columns[0])[0].idxmax().values
        
        # Set matching attribute
        self.matching_ = matching.take(indices).drop(columns=0).reset_index(drop=True)

        # Merge data
        spi_data = self._merge_data(spi_data)
        
        return spi_data

    def _check_leagues_ids(self):
        """Check the leagues ids."""
        if self.leagues_ids != 'all' and not set(LEAGUES_MAPPING.keys()).issuperset(self.leagues_ids):
            error_msg = 'League id should be equal to `all` or a list that contains any of %s elements. Got %s instead.'
            raise ValueError(error_msg % (', '.join(LEAGUES_MAPPING.keys()), self.leagues_ids))
        self.leagues_ids_ = list(LEAGUES_MAPPING.keys()) if self.leagues_ids == 'all' else self.leagues_ids[:]

    def _fetch_data(self, data_type):
        """Download and transform data sources."""
        
        # Check leagues ids
        if not hasattr(self, 'leagues_ids_'):
            self._check_leagues_ids()

        # FD training data
        if not hasattr(self, 'fd_training_data_'):
            self.fd_training_data_ = pd.concat([
                FDTrainingDataSource(league_id, season).download_transform() 
                for league_id, season in tqdm(list(product(self.leagues_ids_, self.seasons)), desc='Downloading')
            ])

        # SPI data
        if not hasattr(self, 'spi_training_data_') and not hasattr(self, 'spi_fixtures_data_'):
            self.spi_training_data_, self.spi_fixtures_data_ = SPIDataSource(self.leagues_ids_).download_transform()

        # FD fixtures data
        if not hasattr(self, 'fd_fixtures_data_') and data_type=='fixtures':
            self._check_leagues_ids()
            self.fd_fixtures_data_ = FDFixturesDataSource(self.leagues_ids_).download_transform()
    
    def _filter_missing_values(self, data):
        """Filter missing values from input data and odds."""
        data = data.dropna(subset=self.input_data_cols, how='any')
        for ind in range(3):
            data = data.dropna(subset=self.odds_cols[ind::3], how='all')
        data.reset_index(drop=True, inplace=True)
        return data

    def _extract_input_data(self, data):
        """Extract input data."""
        X = data.loc[:, self.input_data_cols]
        X['diff_proj_score'] = X['proj_score1'] - X['proj_score2']
        X['diff_spi'] = X['spi1'] - X['spi2']
        X['diff_prob'] = X['prob1'] - X['prob2']
        return X

    def _extract_target(self, data, data_type):
        """Extract target."""
        if data_type == 'fixtures':
            return None
        if self.betting_type == 'MO':
            y = (data['FTHG'] - data['FTAG']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A')
        elif 'OU' in self.betting_type:
            y = (data['FTHG'] + data['FTAG'] > float(self.betting_type[2:])).apply(lambda sign: 'O' if sign > 0 else 'U')
        return y

    def _extract_odds(self, data):
        """Extract maximum odds."""
        odds = data.loc[:, self.odds_cols]
        for ind in range(3):
            odds.loc[:, odds.columns[ind][-1]] = np.nanmax(odds[odds.columns[ind::3]], axis=1)
        odds.drop(columns=self.odds_cols, inplace=True)
        return odds

    def _data(self, data_type):
        """Generate training or fixtures data."""

        # Fetch data
        self._fetch_data(data_type)

        # Rename teams
        spi_data = self._rename_teams(self.spi_training_data_ if data_type == 'training' else self.spi_fixtures_data_)

        # Combine data
        if data_type == data_type == 'training':
            data = pd.merge(spi_data, self.fd_training_data_, left_on=SPIDataSource.match_cols + SPIDataSource.full_goals_cols, right_on=FDDataSource.match_cols + FDDataSource.full_goals_cols)
            data = data[self.match_cols + self.input_data_cols + self.goals_cols + self.odds_cols]
        elif data_type == data_type == 'fixtures':
            data = pd.merge(spi_data, self.fd_fixtures_data_, left_on=SPIDataSource.match_cols, right_on=FDDataSource.match_cols)
            data = data[self.match_cols[1:] + self.input_data_cols + self.odds_cols]
        
        # Filter missing values
        data = self._filter_missing_values(data)

        # Extract input data
        X = self._extract_input_data(data)

        # Extract target
        y = self._extract_target(data, data_type)
        
        # Extract odds
        odds = self._extract_odds(data)

        # Extract matches
        matches = data.loc[:, self.match_cols[int(data_type == 'fixtures'):]]

        return X, y, odds, matches

    @property
    def training_data(self):
        """Generate the training_data."""

        super(SoccerDataLoader, self).training_data
        
        return self._data('training')

    @property
    def fixtures_data(self):
        """Generate the fixtures data."""

        super(SoccerDataLoader, self).fixtures_data
        
        return self._data('fixtures')
