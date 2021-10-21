#!/usr/bin/env python3

import os
import pandas as pd
import streamlit as st
import plotly.express as px


def read_data():
    """Read in all of the CSVs from the working directory."""

    # Read in the tables
    data = {
        fp: pd.read_csv(fp, index_col=0)
        for fp in os.listdir('.')
        if fp.endswith(".csv") or fp.endswith(".csv.gz")
    }

    # Only keep numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = {
        k: df.select_dtypes(include=numerics)
        for k, df in data.items()
    }

    # Make sure that each index is unique
    for table_path, df in data.items():

        n_unique_indices = len(set(df.index.values))
        n_rows = df.shape[0]
        msg = f"Table ({table_path}) may only contain unique values in the first column."
        assert n_unique_indices == n_rows, msg

    return data



def display_sidebar(data):
    """Capture all of the inputs in a single dict."""

    params = dict()

    # Major plot type
    params['type'] = st.sidebar.selectbox(
        "Display Type",
        ["Distribution"]
    )

    # Primary data source
    params['primary_table'] = st.sidebar.selectbox(
        "Showing Data From",
        data.keys()
    )

    params['primary_col'] = st.sidebar.selectbox(
        "Column:",
        data[params['primary_table']].columns.values
    )

    return params


def display_plot(params, data):
    """Display the plot."""

    if params['type'] == 'Distribution':
        display_plot_distribution(params, data)


def display_plot_distribution(params, data):

    df = data[params['primary_table']].sort_values(
        by=params['primary_col']
    )
    
    df = df.assign(
        rank_order=range(df.shape[0])
    ).reset_index()

    print(df.columns.values[0])

    st.write(
        px.histogram(
            df,
            x=params['primary_col']
        )
    )

    st.write(
        px.scatter(
            df,
            y='rank_order',
            x=params['primary_col'],
            hover_data=[df.columns.values[0], params['primary_col']]
        )
    )


def display_app(data):
    """Render the eda-viz app."""

    # Get the params from the user based on inputs in the sidebar
    params = display_sidebar(data)

    # Use the values selected in the sidebar to render the display
    display_plot(params, data)


# READ THE INPUT DATA
data = read_data()

# DISPLAY THE APP
display_app(data)
