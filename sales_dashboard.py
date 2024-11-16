# Required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Simulated data for the dashboard
data = {
    'sales_id': range(1, 1001),
    'order_id': range(10000, 11000),
    'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004'] * 250,
    'product_name': ['Product A', 'Product B', 'Product C', 'Product D'] * 250,
    'order_date': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
    'sales': np.random.uniform(50, 1000, 1000).round(2),
    'region': ['North', 'South', 'East', 'West'] * 250
}

# Load data into a pandas DataFrame
sales_data = pd.DataFrame(data)

# Dashboard title
st.title("Comprehensive Sales Analytics Dashboard")

# Sidebar for filter options
st.sidebar.header("Filter Options")
date_range = st.sidebar.date_input(
    "Select date range", 
    [sales_data['order_date'].min(), sales_data['order_date'].max()]
)

# Convert date inputs to pandas Timestamp objects
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Ensure valid date range selection
if start_date > end_date:
    st.error("Start date cannot be after end date. Please choose a valid date range.")
else:
    category_filter = st.sidebar.multiselect(
        "Select Product Category", 
        sales_data['product_name'].unique(), 
        default=sales_data['product_name'].unique()
    )
    region_filter = st.sidebar.multiselect(
        "Select Region", 
        sales_data['region'].unique(), 
        default=sales_data['region'].unique()
    )

    # Filter data based on user input
    filtered_data = sales_data[
        (sales_data['order_date'].between(start_date, end_date)) &
        (sales_data['product_name'].isin(category_filter)) &
        (sales_data['region'].isin(region_filter))
    ]

    # Display key metrics
    st.header("Key Metrics")

    col1, col2, col3 = st.columns(3)

    # Total Sales
    col1.metric("Total Sales", f"${filtered_data['sales'].sum():,.2f}")
    
    # Number of Orders
    col2.metric("Total Orders", f"{filtered_data['order_id'].nunique()}")

    # Average Sales per Order
    avg_sales = filtered_data['sales'].mean() if not filtered_data.empty else 0
    col3.metric("Average Sales per Order", f"${avg_sales:,.2f}")

    if not filtered_data.empty:
        # Top-selling Product
        top_product = filtered_data.groupby('product_name')['sales'].sum().idxmax()
        st.metric("Top-selling Product", top_product)

        # Top Region by Sales
        top_region = filtered_data.groupby('region')['sales'].sum().idxmax()
        st.metric("Top Region", top_region)
    else:
        st.warning("No data available for the selected filters.")

    # Sales Visualizations
    st.header("Sales Visualizations")

    if not filtered_data.empty:
        # Sales by Product
        st.subheader("Sales by Product")
        sales_by_product = filtered_data.groupby('product_name')['sales'].sum().reset_index()
        fig_product = px.bar(sales_by_product, x='product_name', y='sales', title='Sales by Product')
        st.plotly_chart(fig_product)

        # Sales Over Time
        st.subheader("Sales Over Time")
        sales_over_time = filtered_data.groupby('order_date')['sales'].sum().reset_index()
        fig_time = px.line(sales_over_time, x='order_date', y='sales', title='Sales Over Time')
        st.plotly_chart(fig_time)

        # Sales by Region
        st.subheader("Sales by Region")
        sales_by_region = filtered_data.groupby('region')['sales'].sum().reset_index()
        fig_region = px.pie(sales_by_region, values='sales', names='region', title='Sales by Region')
        st.plotly_chart(fig_region)

        # Sales Distribution (Histogram)
        st.subheader("Sales Distribution")
        fig_hist = px.histogram(filtered_data, x='sales', nbins=20, title='Sales Distribution')
        st.plotly_chart(fig_hist)

    else:
        st.warning("No data to visualize for the selected filters.")
