import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import leastsq
import numpy as np

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D = p
    return logistic4(x, A, B, C, D)

def least_squares_fit(xdata, ydata):

    # Initial guess for parameters
    p0 = [0, 1, 50, 0.05]

    # Fit equation using least squares optimization
    plsq = leastsq(residuals, p0, args=(ydata, xdata))

    # residual sum of squares
    ss_res = np.sum((ydata - peval(xdata,plsq[0])) ** 2)
    # total sum of squares
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return plsq, r2

def logistic4(x, A, B, C, D):
    """4PL lgoistic equation.
    where 
    (x) is the concentration, 
    (A) is the minimum asymptote, 
    (B) is the steepness, 
    (C) is the inflection point
    (D) is the maximum asymptote
    """
    result = np.zeros_like(x)
    valid_indices = (C != 0) & (x/C > 0)  # Only compute for values where (x/C) is positive
    result[valid_indices] = ((A-D)/(1.0+((x[valid_indices]/C)**B))) + D
    return result


def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A,B,C,D = p
    err = y-logistic4(x, A, B, C, D)
    return err

st.set_page_config(layout="wide")

@st.cache_data
def get_sensorgram_data(file):
    return pd.read_csv(file)

@st.cache_data
def get_std_curve_data(file):
    return pd.read_csv(file)

df1 = get_sensorgram_data("sensorgram.csv")
df2 = get_std_curve_data("std_curve.csv")


# Extract all unique categories (based on the conc_string column from df2)
categories = df2['conc_string'].unique()

with st.sidebar:
    with st.expander("Concentration Spikes"):
        selected_categories = [cat for cat in categories if st.checkbox(cat, value=True)]


tab1, tab2, tab3, tab4 = st.tabs(["Standard Curve","Std Curve Sensorgram", "Calibration Spikes", "Measurement Spikes"])


with tab1:
    # Filter df2 based on selected categories
    filtered_df2 = df2[df2['conc_string'].isin(selected_categories)]
    fig2 = px.scatter(filtered_df2, x='concentration', y='max_delta_lambda', color='conc_string', title='Standard Curve')
    fig2.update_layout(showlegend=False, height=600, width=1000)
    fig2.update_traces(marker=dict(size=8, symbol="x"))
    fig2.update_xaxes(type='log')

    xdata = filtered_df2["concentration"].to_numpy()
    ydata = filtered_df2["max_delta_lambda"].to_numpy()

    pt1, pt2 = least_squares_fit(xdata, ydata)
    plsq, _ = pt1
    r2 = pt2
    standard_curve_least_sq = plsq
    standard_curve_r2 = r2
    x_upsampled = np.arange(min(xdata), max(xdata), 0.01)
    y_upsampled = peval(x_upsampled,plsq)

    fig2.add_traces(go.Scatter(x=x_upsampled, y=y_upsampled, mode='lines', name='Fitted Curve', line=dict(color='red')))
    fig2.update_xaxes(type='log')
    st.plotly_chart(fig2)


with st.sidebar:
    df_std_curve_params = pd.DataFrame({"Metric": ["A","B","C","D","R2"], "Value": np.append(plsq, r2)})
    st.dataframe(df_std_curve_params)



with tab2:

    # Create a combined scatter plot for all selected categories
    fig1 = go.Figure()

    for cat in selected_categories:
        time_col = f"Time_{cat}"
        value_col = cat
        fig1.add_trace(go.Scatter(x=df1[time_col], y=df1[value_col].rolling(3).median(), mode='markers', name=cat))

    fig1.update_layout(title="Sensorgram Data", xaxis_title="Time", yaxis_title="Value", showlegend=False, height=600, width=1000)
    st.plotly_chart(fig1)


with st.sidebar:
    df_inferred_conc = pd.DataFrame({"Spike": [1, 2, 3], "Inferred Conc.": [100,101,99]})
    st.dataframe(df_inferred_conc)


# st.divider()




