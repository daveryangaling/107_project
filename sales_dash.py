import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import for date formatting
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
    st.write("This summary table provides a statistical overview of the filtered sales data, including metrics such as mean, median, standard deviation, and percentiles.")
    st.write(filtered_sales.describe())

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
fig, ax = plt.subplots()
ax.plot(filtered_sales['order_date'], filtered_sales['sales'], label="Monthly Sales", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")
st.write("This plot shows the trend of monthly sales over the selected date range. It helps in identifying patterns, seasonality, and any anomalies in the sales data.")

# Apply improved date formatting for better readability
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as "Month Year"
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
plt.xticks(rotation=45)  # Rotate labels for clarity
st.pyplot(fig)

# Forecasting
st.sidebar.subheader("Forecasting Options")
if st.sidebar.checkbox("Show Sales Forecasting"):
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
    model, mae, rmse, test_dates, y_pred, future_dates, future_sales = sales_forecasting(monthly_sales, forecast_horizon)
    
    # Display metrics
    st.subheader("Sales Forecasting")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write("MAE and RMSE are common metrics used to evaluate the accuracy of the forecasting model. Lower values indicate a better fit between the model's predictions and actual sales data.")
    st.write("This section presents the forecasted sales for the next months, along with the accuracy metrics of the model used.")

    # Forecast plot
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['order_date'], monthly_sales['sales'], label="Actual Sales", color='blue')
    ax.plot(test_dates, y_pred, label="Predicted Sales (Test)", color='red', linestyle='--')
    ax.plot(future_dates, future_sales, label="Future Forecast", color='green', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Sales Forecasting")
    ax.legend()
    st.write("The green dashed line represents the sales forecast for the future. This plot helps in visualizing how the sales are expected to trend based on historical data.")

    # Apply improved date formatting for better readability
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as "Month Year"
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
    plt.xticks(rotation=45)
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
    st.write("This plot illustrates the average sales for each month. It helps in understanding seasonal patterns and identifying peak sales periods, which can be useful for planning and decision-making.")
    st.pyplot(fig)

# Data Visualization and Explanation Feature
st.sidebar.subheader("Data Visualization and Explanation")
if st.sidebar.checkbox("Show Data Visualization and Explanation"):
    st.subheader("Detailed Data Visualization and Explanation")
    
    # Explanation of the dataset
    st.write("### Dataset Overview")
    st.write("This dataset contains sales data with order dates and sales figures. Each record represents the total sales made on a particular day. The data is aggregated monthly for better analysis and forecasting.")
    
    # Description of Monthly Sales Trend
    st.write("### Monthly Sales Trend")
    st.write("The Monthly Sales Trend plot shows the trend of sales over time. It helps identify patterns, such as seasonality or trends, and detect any anomalies in the sales data. By analyzing this trend, businesses can understand their sales performance and make informed decisions.")
    
    # Description of Forecasting Model
    st.write("### Sales Forecasting Model")
    st.write("The Sales Forecasting Model uses linear regression to predict future sales based on historical data. The model is trained on past sales data and evaluated using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). These metrics indicate the accuracy of the model—the lower the values, the better the predictions.")
    st.write("The forecast plot shows the actual sales, predicted sales for the test data, and future sales forecast. This helps businesses anticipate future sales and plan accordingly.")
    
    # Explanation of Seasonal Analysis
    st.write("### Seasonal Sales Analysis")
    st.write("The Seasonal Sales Analysis plot illustrates the average sales for each month. This analysis helps identify seasonal patterns in sales, such as higher sales in certain months due to holidays or events. Understanding these patterns can assist in inventory management and marketing strategies.")

# Download Data Option
st.sidebar.subheader("Download Options")
if st.sidebar.button("Download Filtered Data"):
    csv = filtered_sales.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, mime="text/csv")

st.write("Select options in the sidebar to view additional analyses.")
