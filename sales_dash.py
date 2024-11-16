import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Database connection
engine = create_engine('postgresql://postgres:admin@localhost:5432/is107')

# Load data from database
@st.cache_data
def load_data():
    sales_data = pd.read_sql('SELECT order_date, sales FROM fact_sales', engine)
    sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
    return sales_data

# Monthly sales aggregation for forecasting
def prepare_monthly_sales(sales_data):
    monthly_sales = sales_data.groupby(sales_data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()
    return monthly_sales

# Sales forecasting model
def sales_forecasting(monthly_sales, horizon):
    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y = monthly_sales['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Forecast for custom horizon
    future_X = np.arange(len(monthly_sales), len(monthly_sales) + horizon).reshape(-1, 1)
    future_sales = model.predict(future_X)
    future_dates = pd.date_range(monthly_sales['order_date'].max(), periods=horizon, freq='M')

    # Test dates for actual vs predicted
    test_dates = monthly_sales['order_date'].iloc[X_test.flatten()]
    return model, mae, rmse, test_dates, y_pred, future_dates, future_sales

# Main app layout
st.title("Sales Data Dashboard")
st.sidebar.title("Dashboard Options")

# Load data
sales_data = load_data()
monthly_sales = prepare_monthly_sales(sales_data)

# Date range filter
st.sidebar.subheader("Filter by Date Range")
start_date = st.sidebar.date_input("Start Date", monthly_sales['order_date'].min())
end_date = st.sidebar.date_input("End Date", monthly_sales['order_date'].max())

# Convert start_date and end_date to datetime to match 'order_date' format
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the data using the adjusted date range
filtered_sales = monthly_sales[(monthly_sales['order_date'] >= start_date) & (monthly_sales['order_date'] <= end_date)]

# Show data summary
if st.sidebar.checkbox("Show Data Summary"):
    st.subheader("Sales Data Summary")
    st.write(filtered_sales.describe())

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
fig, ax = plt.subplots()
ax.plot(filtered_sales['order_date'], filtered_sales['sales'], label="Monthly Sales", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")
st.pyplot(fig)

# Forecasting
st.sidebar.subheader("Forecasting Options")
if st.sidebar.checkbox("Show Sales Forecasting"):
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
    model, mae, rmse, test_dates, y_pred, future_dates, future_sales = sales_forecasting(monthly_sales, forecast_horizon)
    
    # Display metrics
    st.subheader("Sales Forecasting")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Forecast plot
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['order_date'], monthly_sales['sales'], label="Actual Sales", color='blue')
    ax.plot(test_dates, y_pred, label="Predicted Sales (Test)", color='red', linestyle='--')
    ax.plot(future_dates, future_sales, label="Future Forecast", color='green', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Forecasting")
    ax.legend()
    st.pyplot(fig)

    # Calculate forecast errors if user opts for it
    if st.sidebar.checkbox("Show Forecast Error Distribution"):
        st.subheader("Forecast Error Distribution")
        errors = y_pred - monthly_sales['sales'].iloc[test_dates.index]
        
        fig, ax = plt.subplots()
        ax.hist(errors, bins=10, color='purple', edgecolor='black')
        ax.set_xlabel("Forecast Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Forecast Error Distribution")
        st.pyplot(fig)

# Additional Forecasting Model
if st.sidebar.checkbox("Compare with Exponential Smoothing"):
    st.subheader("Exponential Smoothing Forecast")
    exp_model = ExponentialSmoothing(monthly_sales['sales'], trend="add", seasonal=None, initialization_method="estimated").fit()
    monthly_sales['exp_forecast'] = exp_model.fittedvalues
    
    # Exponential smoothing plot
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['order_date'], monthly_sales['sales'], label="Actual Sales", color='blue')
    ax.plot(monthly_sales['order_date'], monthly_sales['exp_forecast'], label="Exponential Smoothing Forecast", color='orange', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Exponential Smoothing Forecasting")
    ax.legend()
    st.pyplot(fig)

# Seasonal Analysis
if st.sidebar.checkbox("Show Seasonal Analysis"):
    st.subheader("Seasonal Sales Analysis")
    monthly_sales['Month'] = monthly_sales['order_date'].dt.month
    seasonal_sales = monthly_sales.groupby('Month')['sales'].mean()
    
    # Seasonal plot
    fig, ax = plt.subplots()
    ax.plot(seasonal_sales.index, seasonal_sales.values, marker='o', linestyle='-', color='purple')
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")
    ax.set_title("Seasonal Sales Analysis")
    st.pyplot(fig)

# Download Data Option
st.sidebar.subheader("Download Options")
if st.sidebar.button("Download Filtered Data"):
    csv = filtered_sales.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, mime="text/csv")

st.write("Select options in the sidebar to view additional analyses.")
