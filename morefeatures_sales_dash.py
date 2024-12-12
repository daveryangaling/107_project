import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import zscore

# Database connection
engine = create_engine('postgresql://postgres:admin@localhost:5432/is107')

@st.cache_data
def load_data():
    sales_data = pd.read_sql('SELECT order_date, sales FROM fact_sales', engine)
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    return sales_data

def prepare_monthly_sales(sales_data):
    monthly_sales = sales_data.groupby(sales_data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()
    return monthly_sales

def detect_anomalies(data, threshold=3):
    data['zscore'] = zscore(data['sales'])
    anomalies = data[np.abs(data['zscore']) > threshold]
    return anomalies

st.title("Enhanced Sales Data Dashboard")
st.sidebar.title("Dashboard Options")

# Load data
sales_data = load_data()
monthly_sales = prepare_monthly_sales(sales_data)

# Date Range Filter
st.sidebar.subheader("Filter by Date Range")
start_date = st.sidebar.date_input("Start Date", monthly_sales['order_date'].min())
end_date = st.sidebar.date_input("End Date", monthly_sales['order_date'].max())

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_sales = monthly_sales[(monthly_sales['order_date'] >= start_date) & (monthly_sales['order_date'] <= end_date)]

# KPI Metrics
st.sidebar.subheader("Key Metrics")
total_sales = filtered_sales['sales'].sum()
avg_sales = filtered_sales['sales'].mean()
growth_rate = (filtered_sales['sales'].iloc[-1] - filtered_sales['sales'].iloc[0]) / filtered_sales['sales'].iloc[0] * 100

st.sidebar.metric("Total Sales", f"${total_sales:,.2f}")
st.sidebar.metric("Average Monthly Sales", f"${avg_sales:,.2f}")
st.sidebar.metric("Growth Rate", f"{growth_rate:.2f}%")

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
fig = px.line(filtered_sales, x="order_date", y="sales", title="Monthly Sales Trend", markers=True)
st.plotly_chart(fig)

# Anomaly Detection
if st.sidebar.checkbox("Show Anomalies"):
    st.subheader("Anomalies in Sales Data")
    anomalies = detect_anomalies(filtered_sales)
    fig = px.scatter(filtered_sales, x="order_date", y="sales", title="Sales with Anomalies Highlighted")
    fig.add_scatter(x=anomalies['order_date'], y=anomalies['sales'], mode='markers', marker=dict(color='red', size=10), name="Anomalies")
    st.plotly_chart(fig)

# Forecasting Options
st.sidebar.subheader("Forecasting Options")
if st.sidebar.checkbox("Enable Forecasting"):
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
    model = ExponentialSmoothing(filtered_sales['sales'], trend="add", seasonal=None, initialization_method="estimated").fit()
    future_dates = pd.date_range(filtered_sales['order_date'].max(), periods=forecast_horizon, freq='M')
    future_forecast = model.forecast(forecast_horizon)
    
    st.subheader("Sales Forecasting")
    forecast_df = pd.DataFrame({'order_date': future_dates, 'forecasted_sales': future_forecast})
    forecast_fig = px.line(filtered_sales, x="order_date", y="sales", title="Sales Forecast")
    forecast_fig.add_scatter(x=future_dates, y=future_forecast, mode='lines', name='Forecast', line=dict(color='orange'))
    st.plotly_chart(forecast_fig)

# Seasonal Analysis
if st.sidebar.checkbox("Seasonal Analysis"):
    st.subheader("Seasonal Sales Analysis")
    seasonal_sales = monthly_sales.groupby(monthly_sales['order_date'].dt.month)['sales'].mean()
    seasonal_fig = px.bar(seasonal_sales, x=seasonal_sales.index, y=seasonal_sales.values, labels={'x': 'Month', 'y': 'Average Sales'}, title="Seasonal Sales Trends")
    st.plotly_chart(seasonal_fig)

# Download Filtered Data
if st.sidebar.button("Download Filtered Data"):
    csv = filtered_sales.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, mime="text/csv")
