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


def generate_names_mapping(left_data, right_data):
    """Generate names mapping."""

    # Rename columns
    key_columns = ['key%s' % ind for ind in range(left_data.shape[1] - 2)]
    left_data.columns = key_columns + ['left_team1', 'left_team2']
    right_data.columns = key_columns + ['right_team1', 'right_team2']

    # Generate teams names combinations
    names_combinations = pd.merge(left_data, right_data, how='outer').dropna().drop(columns=key_columns).reset_index(drop=True)

    # Calculate similarity index
    similarity = names_combinations.apply(lambda row: SequenceMatcher(None, row.left_team1, row.right_team1).ratio() * SequenceMatcher(None, row.left_team2, row.right_team2).ratio(), axis=1)

    # Append similarity index
    names_combinations_similarity = pd.concat([names_combinations, similarity], axis=1)

    # Filter correct matches
    indices = names_combinations_similarity.groupby(['left_team1', 'left_team2'])[0].idxmax().values
    names_matching = names_combinations.take(indices)

    # Teams matching
    matching1 = names_matching.loc[:, ['left_team1', 'right_team1']].rename(columns={'left_team1': 'left_team', 'right_team1': 'right_team'})
    matching2 = names_matching.loc[:, ['left_team2', 'right_team2']].rename(columns={'left_team2': 'left_team', 'right_team2': 'right_team'})
        
    # Combine matching
    matching = matching1.append(matching2)
    matching = matching.groupby(matching.columns.tolist()).size().reset_index()
    indices = matching.groupby('left_team')[0].idxmax().values
        
    # Generate mapping
    mapping = matching.take(indices).drop(columns=0).reset_index(drop=True)
        
    return mapping


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
        content = [content[mask].reset_index(drop=True), content[~mask].reset_index(drop=True)]

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
        content = self.content_.reset_index(drop=True).copy()

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
        content = content.loc[content['Div'].isin(self.leagues_ids)].reset_index(drop=True)

        return content


class SoccerDataLoader(BaseDataLoader):

    seasons = ['1617', '1718', '1819']
    match_cols = ['Season'] + FDDataSource.match_cols
    input_data_cols = FDDataSource.avg_max_odds_cols + SPIDataSource.spi_cols
    goals_cols = FDDataSource.full_goals_cols + FDDataSource.half_goals_cols
    odds_cols = FDDataSource.odds_cols

    def __init__(self, leagues_ids, target_type):
        
        # Check leagues ids
        leauges_ids_error_msg = 'Parameter `leagues_ids` should be equal to `all` or a list that contains any of %s elements. Got %s instead.' % (', '.join(LEAGUES_MAPPING.keys()), leagues_ids)
        if not isinstance(leagues_ids, (str, list)):
            raise TypeError(leauges_ids_error_msg)
        if leagues_ids != 'all' and not set(LEAGUES_MAPPING.keys()).issuperset(leagues_ids):
            raise ValueError(leauges_ids_error_msg)
        self.leagues_ids_ = list(LEAGUES_MAPPING.keys()) if leagues_ids == 'all' else leagues_ids[:]
        
        # Check betting type
        target_type_error_msg = 'Wrong target type.'
        if not isinstance(target_type, str):
            raise TypeError(target_type_error_msg)
        if target_type not in ('full_time_results', 'half_time_results', 'both_score') and 'over' not in target_type and 'under' not in target_type:
            raise ValueError(target_type_error_msg)
        self.target_type_ = target_type

    def _fetch_data(self, data_type):
        """Download and transform data sources."""

        # FD training data
        if not hasattr(self, 'fd_training_data_'):
            self.fd_training_data_ = pd.concat([
                FDTrainingDataSource(league_id, season).download_transform() 
                for league_id, season in tqdm(list(product(self.leagues_ids_, self.seasons)), desc='Downloading')
            ], ignore_index=True)

        # SPI data
        if not hasattr(self, 'spi_training_data_') and not hasattr(self, 'spi_fixtures_data_'):
            self.spi_training_data_, self.spi_fixtures_data_ = SPIDataSource(self.leagues_ids_).download_transform()

        # FD fixtures data
        if not hasattr(self, 'fd_fixtures_data_') and data_type=='fixtures':
            self.fd_fixtures_data_ = FDFixturesDataSource(self.leagues_ids_).download_transform()

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
        if self.target_type_ == 'full_time_results':
            y = (data['FTHG'] - data['FTAG']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A')
        if self.target_type_ == 'half_time_results':
            y = (data['HTHG'] - data['HTAG']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A')
        elif 'over' in self.target_type_:
            y = (data['FTHG'] + data['FTAG'] > float(self.target_type_[-2:])).astype(int)
        elif 'under' in self.target_type_:
            y = (data['FTHG'] + data['FTAG'] < float(self.target_type_[-2:])).astype(int)
        elif 'both_score' in self.target_type_:
            y = (data['FTHG'] * data['FTAG'] > 0).astype(int)
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
        if not hasattr(self, 'mapping_'):
            self.mapping_ = generate_names_mapping(self.spi_training_data_.loc[:, SPIDataSource.match_cols], self.fd_training_data_.loc[:, FDDataSource.match_cols])
        spi_data = self.spi_training_data_ if data_type == 'training' else self.spi_fixtures_data_
        for col in ['team1', 'team2']:
            spi_data = pd.merge(spi_data, self.mapping_, left_on=col, right_on='left_team', how='left').drop(columns=[col, 'left_team']).rename(columns={'right_team': col})

        # Combine data
        if data_type == 'training':
            data = pd.merge(spi_data, self.fd_training_data_, left_on=SPIDataSource.match_cols + SPIDataSource.full_goals_cols, right_on=FDDataSource.match_cols + FDDataSource.full_goals_cols)
            data = data.loc[:, self.match_cols + self.input_data_cols + self.goals_cols + self.odds_cols]
        elif data_type == 'fixtures':
            data = pd.merge(spi_data, self.fd_fixtures_data_, left_on=SPIDataSource.match_cols, right_on=FDDataSource.match_cols)
            data = data.loc[:, self.match_cols[1:] + self.input_data_cols + self.odds_cols]
        
        # Filter missing values
        data = data.dropna(subset=self.input_data_cols, how='any')
        for ind in range(3):
            data = data.dropna(subset=self.odds_cols[ind::3], how='all')
        data.reset_index(drop=True, inplace=True)

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
