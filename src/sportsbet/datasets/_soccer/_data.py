"""Download and transform historical and fixtures data.

Data sources:

- [Football-Data.co.uk](http://www.football-data.co.uk/data.php)
- [FiveThirtyEight](https://github.com/fivethirtyeight/data/tree/master/soccer-spi)
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sklearn.model_selection import ParameterGrid
from typing_extensions import Self

from ... import FixturesData, TrainingData
from .._base import _BaseDataLoader
from ._fd import _FDSoccerDataLoader
from ._fte import _FTESoccerDataLoader
from ._utils import OUTPUTS

NAMES_MAPPING = {
    'SK Austria Klagenfurt': 'A. Klagenfurt',
    'AEK Athens': 'AEK',
    'AaB': 'Aalborg',
    'AGF Aarhus': 'Aarhus',
    'Accrington Stanley': 'Accrington',
    'Adana Demirspor': 'Ad. Demirspor',
    'FC Trenkwalder Admira': 'Admira',
    'AC Ajaccio': 'Ajaccio',
    'GFC Ajaccio': 'Ajaccio GFCO',
    'Akhisar Belediye': 'Akhisar Belediyespor',
    'Terek Grozny': 'Akhmat Grozny',
    'AlavÃ©s': 'Alaves',
    'AD Alcorcon': 'Alcorcon',
    'Cashpoint SC Rheindorf Altach': 'Altach',
    'AmÃ©rica Mineiro': 'America MG',
    'Amkar Perm': 'Amkar',
    'Apollon Smyrni': 'Apollon',
    'Argentinos Juniors': 'Argentinos Jrs',
    'Aris Salonika': 'Aris',
    'FC Arouca': 'Arouca',
    'FC Arsenal Tula': 'Arsenal Tula',
    'Athletic Bilbao': 'Ath Bilbao',
    'Atletico Madrid': 'Ath Madrid',
    'AtlÃ©tico San Luis': 'Atl. San Luis',
    'AtlÃ©tico TucumÃ¡n': 'Atl. Tucuman',
    'Atlanta United FC': 'Atlanta United',
    'Atletico Mineiro': 'Atletico-MG',
    'AtlÃ©tico Paranaense': 'Atletico-PR',
    'FC Augsburg': 'Augsburg',
    'FK Austria Vienna': 'Austria Vienna',
    'AvaÃ\xad': 'Avai',
    'BahÃ\xada': 'Bahia',
    'FC Barcelona II': 'Barcelona B',
    'KFCO Beerschot-Wilrijk': 'Beerschot VA',
    'Guizhou Renhe': 'Beijing Renhe',
    'Belgrano Cordoba': 'Belgrano',
    'Real Betis': 'Betis',
    'Beziers AS': 'Beziers',
    'Arminia Bielefeld': 'Bielefeld',
    'VfL Bochum': 'Bochum',
    'Botafogo': 'Botafogo RJ',
    'Bourg-Peronnas': 'Bourg Peronnas',
    'AFC Bournemouth': 'Bournemouth',
    'Bradford City': 'Bradford',
    'SK Brann': 'Brann',
    'Eintracht Braunschweig': 'Braunschweig',
    'Brighton and Hove Albion': 'Brighton',
    'Bristol Rovers': 'Bristol Rvs',
    'Istanbul Basaksehir': 'Buyuksehyr',
    'Cambridge United': 'Cambridge',
    'Cambuur Leeuwarden': 'Cambuur',
    'Cardiff City': 'Cardiff',
    'Carlisle United': 'Carlisle',
    'FC Cartagena': 'Cartagena',
    'CearÃ¡': 'Ceara',
    'Celta Vigo': 'Celta',
    'Central CÃ³rdoba Santiago del Estero': 'Central Cordoba',
    'Chambly Thelle FC': 'Chambly',
    'Chapecoense AF': 'Chapecoense-SC',
    'Sporting de Charleroi': 'Charleroi',
    'Charlton Athletic': 'Charlton',
    'Cheltenham Town': 'Cheltenham',
    'Chiapas FC': 'Chiapas',
    'Chievo Verona': 'Chievo',
    'Chongqing Lifan': 'Chongqing Liangjiang Athletic',
    'Clermont Foot': 'Clermont',
    'Club AmÃ©rica': 'Club America',
    'Tijuana': 'Club Tijuana',
    'Montreal Impact': 'Club de Foot Montreal',
    'Colchester United': 'Colchester',
    'Colon Santa Fe': 'Colon Santa FE',
    'Coventry City': 'Coventry',
    'Crewe Alexandra': 'Crewe',
    'Dalian Aerbin': 'Dalian Yifang F.C.',
    'Dalkurd FF': 'Dalkurd',
    'SV Darmstadt 98': 'Darmstadt',
    'Degerfors IF': 'Degerfors',
    'ADO Den Haag': 'Den Haag',
    'Derby County': 'Derby',
    'Dijon FCO': 'Dijon',
    'Djurgardens IF': 'Djurgarden',
    'Doncaster Rovers': 'Doncaster',
    'Borussia Dortmund': 'Dortmund',
    'Dynamo Dresden': 'Dresden',
    'MSV Duisburg': 'Duisburg',
    'Dundee Utd': 'Dundee United',
    'Dinamo Moscow': 'Dynamo Moscow',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'IF Elfsborg': 'Elfsborg',
    'Erzurumspor': 'Erzurum BB',
    'Espanyol': 'Espanol',
    'Estoril Praia': 'Estoril',
    'Estudiantes': 'Estudiantes L.P.',
    'Exeter City': 'Exeter',
    'Emmen': 'FC Emmen',
    'FC Cologne': 'FC Koln',
    'Anzhi Makhachkala': 'FK Anzi Makhackala',
    'Krylia Sovetov': 'FK Krylya Sovetov Samara',
    'Rostov': 'FK Rostov',
    'Falkenbergs FF': 'Falkenbergs',
    'SC Farense': 'Farense',
    'Flamengo': 'Flamengo RJ',
    'Fortuna Sittard': 'For Sittard',
    'Forest Green Rovers': 'Forest Green',
    'Fortuna DÃ¼sseldorf': 'Fortuna Dusseldorf',
    'SC Freiburg': 'Freiburg',
    'Gazisehir Gaziantep': 'Gaziantep',
    'KAA Gent': 'Gent',
    'Gimnasia La Plata': 'Gimnasia L.P.',
    'GimnÃ¡stic Tarragona': 'Gimnastic',
    'Girona FC': 'Girona',
    'GoiÃ¡s': 'Goias',
    'IFK Goteborg': 'Goteborg',
    'Goztepe': 'Goztep',
    'De Graafschap': 'Graafschap',
    'Grasshoppers ZÃ¼rich': 'Grasshoppers',
    'GrÃªmio': 'Gremio',
    'SpVgg Greuther FÃ¼rth': 'Greuther Furth',
    'Grimsby Town': 'Grimsby',
    'FC Groningen': 'Groningen',
    'Guadalajara': 'Guadalajara Chivas',
    'Guangzhou RF': 'Guangzhou R&F',
    'Guizhou Hengfeng Zhicheng': 'Guizhou Zhicheng',
    'BK Hacken': 'Hacken',
    'Halmstads BK': 'Halmstad',
    'Hamburg SV': 'Hamburg',
    'Hamilton Academical': 'Hamilton',
    'Hannover 96': 'Hannover',
    'Harrogate Town': 'Harrogate',
    'Hebei China Fortune FC': 'Hebei',
    '1. FC Heidenheim 1846': 'Heidenheim',
    'Helsingborgs IF': 'Helsingborg',
    'Henan Jianye': 'Henan Songshan Longmen',
    'Hertha Berlin': 'Hertha',
    'Hobro IK': 'Hobro',
    'TSG Hoffenheim': 'Hoffenheim',
    'Consadole Sapporo': 'Hokkaido Consadole Sapporo',
    'AC Horsens': 'Horsens',
    'Huddersfield Town': 'Huddersfield',
    'SD Huesca': 'Huesca',
    'Hull City': 'Hull',
    'HuracÃ¡n': 'Huracan',
    'UD Ibiza': 'Ibiza',
    'CA Independiente': 'Independiente',
    'FC Ingolstadt 04': 'Ingolstadt',
    'Internazionale': 'Inter',
    'Inter Miami CF': 'Inter Miami',
    'Ionikos FC': 'Ionikos',
    'Ipswich Town': 'Ipswich',
    'Jubilo Iwata': 'Iwata',
    'Jiangsu Suning FC': 'Jiangsu Suning',
    'Jonkopings Sodra IF': 'Jonkopings',
    'FC JuÃ¡rez': 'Juarez',
    '1. FC Kaiserslautern': 'Kaiserslautern',
    'Kalmar FF': 'Kalmar',
    'KarabÃ¼kspor': 'Karabukspor',
    'Fatih KaragÃ¼mrÃ¼k': 'Karagumruk',
    'Karlsruher SC': 'Karlsruhe',
    'FC Khimki': 'Khimki',
    'KV Kortrijk': 'Kortrijk',
    'FC Krasnodar': 'Krasnodar',
    'Kristiansund BK': 'Kristiansund',
    'LASK Linz': 'LASK',
    'Deportivo La CoruÃ±a': 'La Coruna',
    'Larissa': 'Larisa',
    'Lausanne Sports': 'Lausanne',
    'Leeds United': 'Leeds',
    'Leicester City': 'Leicester',
    'Cultural Leonesa': 'Leonesa',
    'Levadiakos': 'Levadeiakos',
    'Bayer Leverkusen': 'Leverkusen',
    'Lincoln City': 'Lincoln',
    'Lobos de la BUAP': 'Lobos BUAP',
    'KSC Lokeren': 'Lokeren',
    'La Hoya Lorca': 'Lorca',
    'FC Lugano': 'Lugano',
    'Luton Town': 'Luton',
    'FC Luzern': 'Luzern',
    'Borussia Monchengladbach': "M'gladbach",
    '1. FC Magdeburg': 'Magdeburg',
    'MÃ¡laga': 'Malaga',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Mansfield Town': 'Mansfield',
    'SV Mattersburg': 'Mattersburg',
    'MazatlÃ¡n FC': 'Mazatlan FC',
    'KV Mechelen': 'Mechelen',
    'FC Midtjylland': 'Midtjylland',
    'AC Milan': 'Milan',
    'Minnesota United FC': 'Minnesota United',
    'AS Monaco': 'Monaco',
    'Morelia': 'Monarcas',
    'Mouscron-Peruwelz': 'Mouscron',
    'NAC': 'NAC Breda',
    'C.D. Nacional': 'Nacional',
    'Nagoya Grampus Eight': 'Nagoya Grampus',
    'AS Nancy Lorraine': 'Nancy',
    'New York City FC': 'New York City',
    "Newell's Old Boys": 'Newells Old Boys',
    'NEC': 'Nijmegen',
    'FK Nizhny Novgorod': 'Nizhny Novgorod',
    'FC Nordsjaelland': 'Nordsjaelland',
    'IFK Norrkoping': 'Norrkoping',
    'Northampton Town': 'Northampton',
    'Norwich City': 'Norwich',
    'Nottingham Forest': "Nott'm Forest",
    '1. FC NÃ¼rnberg': 'Nurnberg',
    'Odd BK': 'Odd',
    'Odense BK': 'Odense',
    'Oldham Athletic': 'Oldham',
    'Olimpo': 'Olimpo Bahia Blanca',
    'Olympiacos': 'Olympiakos',
    'KV Oostende': 'Oostende',
    'Orebro SK': 'Orebro',
    'Gazovik Orenburg': 'Orenburg',
    'Orlando City SC': 'Orlando City',
    'OrlÃ©ans': 'Orleans',
    'VfL Osnabruck': 'Osnabruck',
    'Ã\x96stersunds FK': 'Ostersunds',
    'OH Leuven': 'Oud-Heverlee Leuven',
    'Real Oviedo': 'Oviedo',
    'Oxford United': 'Oxford',
    'PAOK Salonika': 'PAOK',
    'SC Paderborn': 'Paderborn',
    'ParanÃ¡': 'Parana',
    'Paris Saint-Germain': 'Paris SG',
    'Partick Thistle': 'Partick',
    'Pau': 'Pau FC',
    'US Pescara': 'Pescara',
    'Peterborough United': 'Peterboro',
    'Plymouth Argyle': 'Plymouth',
    'SD Ponferradina': 'Ponferradina',
    'Pordenone Calcio': 'Pordenone',
    'FC Porto': 'Porto',
    'Preston North End': 'Preston',
    'QuerÃ©taro': 'Queretaro',
    'US Quevilly': 'Quevilly Rouen',
    'FK Volgograd': 'R. Volgograd',
    'Red Star FC 93': 'Red Star',
    'Jahn Regensburg': 'Regensburg',
    'Stade Rennes': 'Rennes',
    'SV Ried': 'Ried',
    'Caykur Rizespor': 'Rizespor',
    'Roda JC': 'Roda',
    'AS Roma': 'Roma',
    'Rotherham United': 'Rotherham',
    'Energiya Khabarovsk': 'SKA Khabarovsk',
    'CD Sabadell': 'Sabadell',
    'Salford City': 'Salford',
    'FC Salzburg': 'Salzburg',
    'San Martin San Juan': 'San Martin S.J.',
    'San Martin de Tucuman': 'San Martin T.',
    'SV Sandhausen': 'Sandhausen',
    'Racing Santander': 'Santander',
    'SÃ£o Paulo': 'Sao Paulo',
    'Sarpsborg': 'Sarpsborg 08',
    'Seattle Sounders FC': 'Seattle Sounders',
    'RFC Seraing': 'Seraing',
    'Vitoria Setubal': 'Setubal',
    'Sevilla FC': 'Sevilla',
    'Sevilla Atletico': 'Sevilla B',
    'Shanghai Greenland': 'Shanghai Shenhua',
    'Sheffield Wednesday': 'Sheffield Weds',
    'Shenzhen FC': 'Shenzhen',
    'Shrewsbury Town': 'Shrewsbury',
    'FC Sion': 'Sion',
    'IK Sirius': 'Sirius',
    'Real Sociedad': 'Sociedad',
    'Real Sociedad II': 'Sociedad B',
    'Southend United': 'Southend',
    'Braga': 'Sp Braga',
    'Sporting GijÃ³n': 'Sp Gijon',
    'Sparta': 'Sparta Rotterdam',
    'FC St. Pauli': 'St Pauli',
    'St. Truidense': 'St Truiden',
    'St Gallen': 'St. Gallen',
    'Union Saint Gilloise': 'St. Gilloise',
    'St. PÃ¶lten': 'St. Polten',
    'Standard Liege': 'Standard',
    'IK Start': 'Start',
    'Stoke City': 'Stoke',
    'SK Sturm Graz': 'Sturm Graz',
    'VfB Stuttgart': 'Stuttgart',
    'GIF Sundsvall': 'Sundsvall',
    'Sutton United': 'Sutton',
    'Swansea City': 'Swansea',
    'Swindon Town': 'Swindon',
    'Talleres de CÃ³rdoba': 'Talleres Cordoba',
    'FC Tambov': 'Tambov',
    'Tianjin Quanujian': 'Tianjin Quanjian',
    'Tianjin Teda': 'Tianjin Tianhai',
    'WSG Swarovski Wattens': 'Tirol',
    'Tokushima Vortis': 'Tokushima',
    'FC Tosno': 'Tosno',
    'Tottenham Hotspur': 'Tottenham',
    'Tranmere Rovers': 'Tranmere',
    'Trelleborgs FF': 'Trelleborgs',
    'FC Twente': 'Twente',
    'Tigres UANL': 'U.A.N.L.- Tigres',
    'FC Ufa': 'Ufa',
    '1. FC Union Berlin': 'Union Berlin',
    'Union Santa Fe': 'Union de Santa Fe',
    'Urawa Red Diamonds': 'Urawa Reds',
    'FC Utrecht': 'Utrecht',
    'FC Vaduz': 'Vaduz',
    'Real Valladolid': 'Valladolid',
    'Rayo Vallecano': 'Vallecano',
    'Varbergs BoIS FC': 'Varbergs',
    'Vasco da Gama': 'Vasco',
    'F.B.C Unione Venezia': 'Venezia',
    'Viking FK': 'Viking',
    'VÃ\xadtoria': 'Vitoria',
    'FC Wacker Innsbruck': 'Wacker Innsbruck',
    'SV Zulte Waregem': 'Waregem',
    'SV Wehen Wiesbaden': 'Wehen',
    'West Bromwich Albion': 'West Brom',
    'West Ham United': 'West Ham',
    'Wolfsberger AC': 'Wolfsburg',
    'VfL Wolfsburg': 'Wolfsburg',
    'Wuhan Zall': 'Wuhan FC',
    'WÃ¼rzburger Kickers': 'Wurzburger Kickers',
    'Wycombe Wanderers': 'Wycombe',
    'Neuchatel Xamax': 'Xamax',
    'FC Xanthi': 'Xanthi',
    'Matsumoto Yamaga FC': 'Yamaga',
    'Yeovil Town': 'Yeovil',
    'Real Zaragoza': 'Zaragoza',
    'FC Zurich': 'Zurich',
    'PEC Zwolle': 'Zwolle',
}


class SoccerDataLoader(_BaseDataLoader):
    """Dataloader for soccer data from combining all data sources.

    It downloads historical and fixtures data from
    [Football-Data.co.uk](http://www.football-data.co.uk/data.php) and
    [FiveThirtyEight](https://github.com/fivethirtyeight/data/tree/master/soccer-spi).
    The data are combined in a consistent way.

    Read more in the [user guide][user-guide].

    Args:
        param_grid:
            It selects the type of information that the data include. The keys of
            dictionaries might be parameters like `'league'` or `'division'` while
            the values are sequences of allowed values. It works in a similar way as the
            `param_grid` parameter of the scikit-learn's ParameterGrid class.
            The default value `None` corresponds to all parameters.

    Attributes:
        param_grid_ (ParameterGrid):
            The checked value of parameters grid. It includes all possible parameters if
            `param_grid` is `None`.
        dropped_na_cols_:
            The columns with missing values that are dropped.
        drop_na_thres_:
            The checked value of `drop_na_thres`.
        odds_type_:
            The checked value of `odds_type`.
        input_cols_:
            The columns of `X_train` and `X_fix`.
        output_cols_:
            The columns of `Y_train` and `Y_fix`.
        odds_cols_:
            The columns of `O_train` and `O_fix`.
        target_cols_:
            The columns used for the extraction of output and odds columns.

    Examples:
        >>> from sportsbet.datasets import SoccerDataLoader
        >>> import pandas as pd
        >>> # Get all available parameters to select the training data
        >>> SoccerDataLoader.get_all_params()
        [{'data_source': 'fivethirtyeight', 'division': 1, ...
        >>> # Select only the traning data for the French and Spanish leagues of 2020 year
        >>> dataloader = SoccerDataLoader(
        ... param_grid={'league': ['England', 'Spain'], 'year':[2020]})
        >>> # Get available odds types
        >>> dataloader.get_odds_types()
        [..., 'market_average', ...]
        >>> # Select the market average odds and drop colums with missing values
        >>> X_train, Y_train, O_train = dataloader.extract_train_data(
        ... odds_type='market_average', drop_na_thres=1.0)
        >>> # Odds data include the selected market average odds
        >>> O_train.columns
        Index(['odds__market_average__home_win__full_time_goals', ...
        >>> # Extract the corresponding fixtures data
        >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
        >>> # Training and fixtures input and odds data have the same column names
        >>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
        >>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
        >>> # Fixtures data have always no output
        >>> Y_fix is None
        True
    """

    SCHEMA = (
        _FDSoccerDataLoader.SCHEMA
        + [col for col in _FTESoccerDataLoader.SCHEMA if col not in _FDSoccerDataLoader.SCHEMA]
        + [('data_source', object)]
    )
    OUTPUTS = OUTPUTS

    @classmethod
    @lru_cache
    def _get_full_param_grid(cls: type[SoccerDataLoader]) -> ParameterGrid:
        fd_full_param_grid_df = pd.DataFrame(_FDSoccerDataLoader.get_all_params()).assign(data_source='footballdata')
        fte_full_param_grid_df = pd.DataFrame(_FTESoccerDataLoader.get_all_params()).assign(
            data_source='fivethirtyeight',
        )
        full_param_grid = (
            pd.concat([fd_full_param_grid_df, fte_full_param_grid_df]).drop_duplicates().to_dict('records')
        )
        return ParameterGrid([{name: [val] for name, val in params.items()} for params in full_param_grid])

    @lru_cache  # noqa: B019
    def _get_data(self: Self) -> pd.DataFrame:
        param_grids = {}
        for data_source in ('footballdata', 'fivethirtyeight'):
            param_grids[data_source] = [
                {param: [val] for param, val in params.items() if param != 'data_source'}
                for params in self.param_grid_
                if params['data_source'] == data_source
            ]
        if param_grids['footballdata'] and param_grids['fivethirtyeight']:
            fd_data = (
                _FDSoccerDataLoader(param_grids['footballdata'])._check_param_grid()._validate_data().reset_index()
            )
            fte_data = (
                _FTESoccerDataLoader(param_grids['fivethirtyeight'])._check_param_grid()._validate_data().reset_index()
            )
            for col in ('home_team', 'away_team'):
                fte_data[col] = fte_data[col].apply(lambda name: NAMES_MAPPING.get(name, name))
            return fd_data.merge(fte_data)
        elif param_grids['footballdata'] and not param_grids['fivethirtyeight']:
            fd_data = (
                _FDSoccerDataLoader(param_grids['footballdata'])._check_param_grid()._validate_data().reset_index()
            )
            return fd_data.assign(data_source='footballdata')
        elif not param_grids['footballdata'] and param_grids['fivethirtyeight']:
            fte_data = (
                _FTESoccerDataLoader(param_grids['fivethirtyeight'])._check_param_grid()._validate_data().reset_index()
            )
            for col in ('home_team', 'away_team'):
                fte_data[col] = fte_data[col].apply(lambda name: NAMES_MAPPING.get(name, name))
            return fte_data.assign(data_source='fivethirtyeight')
        else:
            return pd.DataFrame()

    def extract_train_data(
        self: Self,
        drop_na_thres: float = 0.0,
        odds_type: str | None = None,
    ) -> TrainingData:
        """Extract the training data.

        Read more in the [user guide][dataloader].

        It returns historical data that can be used to create a betting
        strategy based on heuristics or machine learning models.

        The data contain information about the matches that belong
        in two categories. The first category includes any information
        known before the start of the match, i.e. the training data `X`
        and the odds data `O`. The second category includes the outcomes of
        matches i.e. the multi-output targets `Y`.

        The method selects only the the data allowed by the `param_grid`
        parameter of the initialization method. Additionally, columns with missing
        values are dropped through the `drop_na_thres` parameter, while the
        types of odds returned is defined by the `odds_type` parameter.

        Args:
            drop_na_thres:
                The threshold that specifies the input columns to drop. It is a float in
                the `[0.0, 1.0]` range. Higher values result in dropping more values.
                The default value `drop_na_thres=0.0` keeps all columns while the
                maximum value `drop_na_thres=1.0` keeps only columns with non
                missing values.

            odds_type:
                The selected odds type. It should be one of the available odds columns
                prefixes returned by the method `get_odds_types`. If `odds_type=None`
                then no odds are returned.

        Returns:
            (X, Y, O):
                Each of the components represent the training input data `X`, the
                multi-output targets `Y` and the corresponding odds `O`, respectively.
        """
        X_train, Y_train, O_train = super().extract_train_data(drop_na_thres, odds_type)
        self.input_cols_ = pd.Index([col for col in self.input_cols_ if col != 'data_source'], dtype=object)
        X_train = X_train.reset_index().drop_duplicates(subset=self.input_cols_)
        return (
            X_train.set_index('date')[self.input_cols_],
            Y_train.take(X_train.index.to_list()),
            O_train.take(X_train.index.to_list()) if O_train is not None else None,
        )

    def extract_fixtures_data(self: Self) -> FixturesData:
        """Extract the fixtures data.

        Read more in the [user guide][dataloader].

        It returns fixtures data that can be used to make predictions for
        upcoming matches based on a betting strategy.

        Before calling the `extract_fixtures_data` method for
        the first time, the `extract_training_data` should be called, in
        order to match the columns of the input, output and odds data.

        The data contain information about the matches known before the
        start of the match, i.e. the training data `X` and the odds
        data `O`. The multi-output targets `Y` is always equal to `None`
        and are only included for consistency with the method `extract_train_data`.

        The `param_grid` parameter of the initialization method has no effect
        on the fixtures data.

        Returns:
            (X, None, O):
                Each of the components represent the fixtures input data `X`, the
                multi-output targets `Y` equal to `None` and the
                corresponding odds `O`, respectively.
        """
        X_fix, Y_fix, O_fix = super().extract_fixtures_data()
        return X_fix, Y_fix, O_fix
