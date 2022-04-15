from enum import unique
from sklearn.model_selection import ParameterGrid
import pandas as pd
import streamlit as st
from sklearn.model_selection import ParameterGrid
from sportsbet.datasets import FDSoccerDataLoader, FTESoccerDataLoader, SoccerDataLoader

DATALOADERS_MAPPING = {
        'Soccer': {
            'All': SoccerDataLoader,
            'Football-Data': FDSoccerDataLoader,
            'FiveThirtyEight': FTESoccerDataLoader,
        }
    }


def get_unique_values(param, all_params_df):
    """Get the parameters grid for a specific parameter."""
    return sorted(all_params_df[param].dropna().unique())


def initialize_attr(name, val):
    """Initialize attribute and value."""
    if name not in st.session_state:
        st.session_state[name] = val

def sport_widget():
    """Widget to select the sport."""
    sport = st.selectbox('Sport', DATALOADERS_MAPPING.keys(), help='Select the sport to download the data.')
    return sport


def source_widget(sport):
    """Widget to select the source."""
    source = st.selectbox('Source', DATALOADERS_MAPPING[sport].keys(), help='Select the data source to download the data.')
    return source


def leagues_widget(all_params_df):
    """Widget to select the leagues."""
    unique_values = get_unique_values('league', all_params_df)
    return st.multiselect('Leagues', unique_values, default=unique_values, help='Select the leagues to include to the training data.')


def divisions_widget(all_params_df):
    """Widget to select the divisions."""
    unique_values = get_unique_values('division', all_params_df)
    min_val, max_val = int(min(unique_values)), int(max(unique_values))
    vals = st.slider('Divisions', min_val, max_val, (min_val, max_val), help='Select the divisions to include to the training data.')
    vals = list(range(vals[0], vals[1] + 1))
    return vals


def years_widget(all_params_df):
    """Widget to select the years."""
    unique_values = get_unique_values('year', all_params_df)
    min_val, max_val = int(min(unique_values)), int(max(unique_values))        
    if min_val < max_val:
        vals = st.slider('Years', min_val, max_val, (min_val, max_val), help='Select the years to include to the training data.')
        vals = list(range(vals[0], vals[1] + 1))
    elif min_val == max_val:
        vals = [min_val]
        st.number_input('Years', min_val, max_val)
    return vals


def append_widget(all_params_df, param_grid):
    """Widget to append parameters grid."""
    clicked = st.button('Append', help='Append selections.')
    if clicked:
        param_grid = {param:vals for param, vals in param_grid.items() if vals}
        param_grid_df = all_params_df.merge(pd.DataFrame(ParameterGrid(param_grid)))
        param_grid_df = pd.concat([pd.DataFrame(ParameterGrid(st.session_state.param_grid)), param_grid_df]).drop_duplicates()
        st.session_state.param_grid = [{param:[val] for param, val in param_grid.items()} for param_grid in param_grid_df.to_dict('records')]


def clear_all_widget():
    """Widget to clear all parameters grid."""
    clicked = st.button('Clear all', help='Clear all selections.')
    if clicked:
        st.session_state.param_grid = []


if __name__ == '__main__':
    
    initialize_attr('param_grid', [])

    st.title('Sports-Betting')
    with st.sidebar:

        st.markdown('# Data')

        # General section
        st.markdown('### Options')
        sport = sport_widget()
        source = source_widget(sport)    
        
        dataloader_class = DATALOADERS_MAPPING[sport][source]
        all_params_df = pd.DataFrame(ParameterGrid(dataloader_class.get_all_params()))
        
        # Filters section
        st.markdown('### Filters')
        leagues = leagues_widget(all_params_df)
        divisions = divisions_widget(all_params_df)
        years = years_widget(all_params_df)

        st.markdown('Actions')
        col1, col2 = st.columns(2)
        with col1:
            append_widget(all_params_df, {'league': leagues, 'division': divisions, 'year': years})
        with col2:
            clear_all_widget()
        
        if st.session_state['param_grid']:
            st.markdown('Applied')
            df = pd.DataFrame((ParameterGrid(st.session_state['param_grid'])))
            df.rename(columns={col: f'{col.title()}s' for col in df.columns}, inplace=True)
            st.dataframe(df)
        dataloader = dataloader_class(param_grid=st.session_state.param_grid if st.session_state.param_grid else None)
        
        st.markdown('### Training')
        clicked = st.button('Download')
        # st.selectbox('Bookmaker', dataloader.get_odds_types())
        if clicked:
            X_train, Y_train, O_train = dataloader.extract_train_data()    
    
    if clicked:        
        st.markdown('### Training data')
        if O_train is None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('**Input**')
                st.dataframe(X_train)
            with col2:
                st.markdown('**Targets**')
                st.dataframe(Y_train)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('**Input**')
                st.dataframe(X_train)
            with col2:
                st.markdown('**Targets**')
                st.dataframe(Y_train)
            with col3:
                st.markdown('**Odds**')
                st.dataframe(O_train)
