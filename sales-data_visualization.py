import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# App Title
st.title("Retail Sales Dashboard")

# Database Connection
@st.cache_resource
def connect_to_database():
    db_url = "postgresql+psycopg2://postgres:admin@localhost/is107"
    engine = create_engine(db_url)
    return engine

def load_data(query, engine):
    """Load data from the database."""
    return pd.read_sql(query, engine)

engine = connect_to_database()

# Query to Load Data
query = """
SELECT 
    fs.sales_id,
    fs.sales AS total_sales,
    fs.order_date,
    dc.customer_name,
    dl.country,
    dp.product_name,
    dp.category
FROM fact_sales fs
JOIN dim_customer dc ON fs.customer_id = dc.customer_id
JOIN dim_location dl ON fs.location_id = dl.location_id
JOIN dim_product dp ON fs.product_id = dp.product_id;
"""
data = load_data(query, engine)

# Sidebar Filters
st.sidebar.header("Filter Options")
data["order_date"] = pd.to_datetime(data["order_date"])

# Date Range Filter
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [data["order_date"].min().date(), data["order_date"].max().date()]
)
if start_date and end_date:
    data = data[(data["order_date"] >= pd.Timestamp(start_date)) & (data["order_date"] <= pd.Timestamp(end_date))]

# Country Filter
country = st.sidebar.selectbox("Select Country", ["All"] + list(data["country"].unique()))
if country != "All":
    data = data[data["country"] == country]

# Category Filter
category = st.sidebar.selectbox("Select Category", ["All"] + list(data["category"].unique()))
if category != "All":
    data = data[data["category"] == category]

# Display Filter Summary
st.markdown(f"""
**Filter Summary**
- Date Range: {start_date} to {end_date}
- Country: {country}
- Category: {category}
""")

# Key Metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sales ($)", f"${data['total_sales'].sum():,.2f}")
with col2:
    st.metric("Total Orders", data["sales_id"].nunique())
with col3:
    st.metric("Unique Customers", data["customer_name"].nunique())

# Visualizations
st.subheader("Visualizations")

# 1. Sales by Category
st.markdown("### Sales by Category")
sales_by_category = data.groupby("category")["total_sales"].sum().reset_index()
fig1 = px.bar(
    sales_by_category, 
    x="category", 
    y="total_sales", 
    title="Sales by Category",
    labels={"category": "Category", "total_sales": "Sales ($)"},
    color="total_sales",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Sales by Country
st.markdown("### Sales by Country")
sales_by_country = data.groupby("country")["total_sales"].sum().reset_index()
fig2 = px.pie(
    sales_by_country, 
    values="total_sales", 
    names="country", 
    title="Sales Distribution by Country"
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Monthly Sales Trends
st.markdown("### Monthly Sales Trends")
data["month"] = data["order_date"].dt.to_period("M").astype(str)
monthly_sales = data.groupby("month")["total_sales"].sum().reset_index()
fig3 = px.line(
    monthly_sales, 
    x="month", 
    y="total_sales", 
    title="Monthly Sales Trends",
    labels={"month": "Month", "total_sales": "Sales ($)"}
)
st.plotly_chart(fig3, use_container_width=True)

# Download Filtered Data
st.markdown("### Download Filtered Data")
csv_data = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Data as CSV",
    data=csv_data,
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
