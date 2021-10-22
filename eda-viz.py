#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from scipy import stats
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
        "Display Type:",
        ["Distribution", "Comparison"]
    )

    # Primary data source
    params['primary_table'] = st.sidebar.selectbox(
        "Showing Data From:",
        data.keys()
    )

    params['primary_col'] = st.sidebar.selectbox(
        "Column:",
        data[params['primary_table']].columns.values
    )

    # If the user has selected a comparison
    if params['type'] == 'Comparison':

        # Compare against which table?
        params['secondary_table'] = st.sidebar.selectbox(
            "Compare Against Data From:",
            data.keys()
        )

        # Compare against everything, or a single column?
        params['secondary_source_type'] = st.sidebar.selectbox(
            "Compare Against:",
            ['All Columns','One Column']
        )

        # If it's a single column
        if params['secondary_source_type'] == "One Column":

            # Get that column
            params['secondary_col'] = st.sidebar.selectbox(
                "Comparison Column:",
                [
                    col_name
                    for col_name in data[params['secondary_table']].columns.values
                    if col_name != params['primary_col']
                ]
            )

    return params


def data_is_aligned(params, data):
    """Check to see if the data is aligned -- if the selected plot requires it."""

    # If they have selected a distribution
    if params['type'] == "Distribution":

        # No need for alignment of row labels
        return True

    # If the user only wants to look at data from one table
    if params['primary_table'] == params['secondary_table']:

        # No need for alignment of row labels
        return True

    # At this point, the user wants to align across two tables
    # Get the number of labels which overlap between the two tables
    overlap = set(data[params['primary_table']].index.values) & set(data[params['secondary_table']].index.values)

    # If there isn't any overlap
    if len(overlap) == 0:

        # Tell the user
        st.write("Both tables must have the same set of row labels in the first (leftmost) column")

        return False

    return True


def display_plot(params, data):
    """Display the plot."""

    # Check to see if the data is aligned (only if it matters for the plot)
    if not data_is_aligned(params, data):
        return

    if params['type'] == 'Distribution':
        display_plot_distribution(params, data)

    elif params['type'] == 'Comparison':
        display_plot_comparison(params, data)


def display_plot_comparison(params, data):

    if params['secondary_source_type'] == "All Columns":

        display_plot_comparison_all(params, data)

    else:
        
        display_plot_comparison_one(params, data)


def display_plot_comparison_all(params, data):
    """Compare a single column from the primary table against all columns from the secondary table."""

    # Get the values from the primary column
    primary_data = data[params['primary_table']][params['primary_col']].dropna()

    # Get the values from the secondary table
    secondary_table = data[params['secondary_table']].reindex(index=primary_data.index.values)

    # Remove any columns which have < 3 observations
    secondary_table = secondary_table.reindex(
        columns=[
            col_name
            for col_name in secondary_table.columns.values
            if secondary_table[col_name].dropna().shape[0] >= 3 and col_name != params['primary_col']
        ]
    )
    
    # If there is no data
    if secondary_table.shape[1] == 0:
        st.write(f"There are no columns in {params['secondary_table']} with data overlapping '{params['primary_col']}' in {params['primary_table']}")
        return

    # Since there is data
    plot_df = pd.DataFrame([
        calc_assoc(col_name, col_data, primary_data)
        for col_name, col_data in secondary_table.items()
    ])

    plot_df = plot_df.assign(
        neg_log10_p=plot_df.neg_log10_p.clip(
            upper=plot_df.neg_log10_p.loc[
                plot_df.neg_log10_p.apply(np.isfinite)
            ].max()
        )
    )

    st.write(
        px.scatter(
            plot_df.rename(
                columns=dict(
                    pearson_r="R (pearson)",
                    neg_log10_p="p-value (-log10)"
                )
            ),
            x="R (pearson)",
            y="p-value (-log10)",
            hover_data=["variable", "p"]
        )
    )

def get_valid_indices(vec):
    return set(
        [
            i
            for i, v in vec.items()
            if v is not None and np.isfinite(v)
        ]
    )


def calc_assoc(col_name, col_data, primary_data):

    # Get the valid indices with finite values for both
    valid_indices = get_valid_indices(col_data) & get_valid_indices(primary_data)

    r, p = stats.pearsonr(
        col_data.loc[valid_indices],
        primary_data.loc[valid_indices]
    )

    return dict(
        variable=col_name,
        pearson_r=r,
        p=p,
        neg_log10_p=-np.log10(p)
    )

def display_plot_comparison_one(params, data):
    """Compare a single column from the primary table against a single column from the secondary table."""

    # Make a DataFrame combining the data
    plot_df = pd.concat([
        data[params['primary_table']].reindex(
            columns=[params['primary_col']]
        ),
        data[params['secondary_table']].reindex(
            columns=[params['secondary_col']]
        ),
    ], axis=1)

    # Get the valid indices with finite values for both
    valid_indices = get_valid_indices(plot_df.iloc[:, 0]) & get_valid_indices(plot_df.iloc[:, 1])
    plot_df = plot_df.loc[valid_indices]

    comparison_stats = calc_assoc(
        params['secondary_col'],
        plot_df.iloc[:, 0],
        plot_df.iloc[:, 1]
    )

    st.write(f"{plot_df.dropna().shape[0]:,} / {plot_df.shape[0]:,} entries have values for both '{params['primary_col']}' and '{params['secondary_col']}'")
    st.write(f"Correlation coefficient: {comparison_stats['pearson_r']}")
    st.write(f"p-value: {comparison_stats['p']}")
    

    # If there is no data with aligned labels
    if plot_df.shape[0] == 0:

        st.write(f"No data could be found for both {params['primary_col']} and {params['secondary_col']}")
        return

    st.write(
        px.scatter(
            plot_df,
            x=params['primary_col'],
            y=params['secondary_col']
        )
    )


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
