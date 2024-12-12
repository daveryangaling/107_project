import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Database Connection
@st.cache_resource
def connect_to_database():
    db_url = "postgresql+psycopg2://postgres:admin@localhost/is107"
    engine = create_engine(db_url)
    return engine

engine = connect_to_database()

# Load Data from Database
@st.cache_data
def load_data_from_db(query):
    return pd.read_sql(query, engine)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a module:", ["Data Visualization", "Data Mining"])

# Data Visualization Module
if app_mode == "Data Visualization":
    st.title("Data Visualization Module")

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
    data = load_data_from_db(query)

    if data is not None:
        st.write("### Dataset Preview")
        st.write(data.head())

        st.write("### Descriptive Statistics")
        st.write(data.describe())

        st.write("### Visualizations")

        # Sales by Category
        st.write("#### Sales by Category")
        st.write("This bar chart shows the total sales for each product category. It's useful for identifying which categories generate the most revenue.")
        sales_by_category = data.groupby('category')['total_sales'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='category', y='total_sales', data=sales_by_category, ax=ax)
        ax.set_title('Sales by Category')
        st.pyplot(fig)

        # Sales by Country
        st.write("#### Sales by Country")
        st.write("This pie chart depicts the distribution of sales across different countries. It helps to visualize the geographical distribution of sales.")
        sales_by_country = data['country'].value_counts().reset_index()
        sales_by_country.columns = ['country', 'count']
        fig, ax = plt.subplots()
        ax.pie(sales_by_country['count'], labels=sales_by_country['country'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Monthly Sales Trends
        st.write("#### Monthly Sales Trends")
        st.write("This line chart illustrates the trend in sales over time, on a monthly basis. It can highlight seasonal trends or patterns in sales.")
        data['order_date'] = pd.to_datetime(data['order_date'])
        data['month_year'] = data['order_date'].dt.to_period('M').astype(str)  # Convert to string
        monthly_sales = data.groupby('month_year')['total_sales'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(x='month_year', y='total_sales', data=monthly_sales, ax=ax)
        ax.set_title('Monthly Sales Trends')
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        st.pyplot(fig)

        # Download Filtered Data
        st.write("### Download Filtered Data")
        st.write("You can download the filtered dataset in CSV format.")
        csv = data.to_csv(index=False)
        st.download_button(label="Download Data as CSV", data=csv, file_name='filtered_data.csv', mime='text/csv')

# Data Mining Module
elif app_mode == "Data Mining":
    st.title("Data Mining: Customer Segmentation & Sales Forecasting")

    # Query for Customer Segmentation
    segmentation_query = """
    SELECT 
        fs.customer_id AS CustomerID,
        SUM(fs.sales) AS TotalSpend,
        COUNT(fs.sales_id) AS PurchaseFrequency,
        MAX(fs.order_date) AS LastPurchaseDate
    FROM public.fact_sales fs
    GROUP BY fs.customer_id;
    """
    customer_data = load_data_from_db(segmentation_query)

    # Ensure data is loaded
    if customer_data is not None:
        customer_data.columns = customer_data.columns.str.lower()
        customer_data["lastpurchasedate"] = pd.to_datetime(customer_data["lastpurchasedate"])
        current_date = customer_data["lastpurchasedate"].max()
        customer_data["recency"] = (current_date - customer_data["lastpurchasedate"]).dt.days

        # Tabs for segmentation and forecasting
        tabs = st.tabs(["Customer Segmentation", "Sales Forecasting"])

        # Tab 1: Customer Segmentation
        with tabs[0]:
            st.subheader("Customer Segmentation with K-Means")
            st.write("This section involves segmenting customers into different clusters based on their spending behavior, purchase frequency, and recency. It's useful for identifying different customer groups for targeted marketing strategies.")
            features = customer_data[["totalspend", "purchasefrequency", "recency"]]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # K-Means Clustering
            k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
            kmeans = KMeans(n_clusters=k, random_state=42)
            customer_data["cluster"] = kmeans.fit_predict(scaled_features)

            # Cluster Summary
            st.subheader("Cluster Summary")
            st.write("This table summarizes the average total spend, purchase frequency, and recency for each customer cluster.")
            cluster_summary = customer_data.groupby("cluster")[["totalspend", "purchasefrequency", "recency"]].mean()
            st.dataframe(cluster_summary)

            # Cluster Visualization
            st.subheader("Cluster Visualization")
            st.write("This scatter plot visualizes the customer clusters based on selected features. It's useful for visually identifying the differences and similarities among the clusters.")
            x_axis = st.selectbox("Select X-axis Feature", ["totalspend", "purchasefrequency", "recency"])
            y_axis = st.selectbox("Select Y-axis Feature", ["recency", "totalspend", "purchasefrequency"], index=1)
            fig_cluster = px.scatter(
                customer_data,
                x=x_axis,
                y=y_axis,
                color="cluster",
                title="Customer Clusters",
                labels={x_axis: x_axis.capitalize(), y_axis: y_axis.capitalize()},
                hover_data=["totalspend", "purchasefrequency", "recency"]
            )
            st.plotly_chart(fig_cluster)

        # Tab 2: Sales Forecasting
        with tabs[1]:
            st.subheader("Sales Forecasting with Linear Regression")
            st.write("This section involves predicting future sales based on historical data using linear regression. It helps in planning and setting sales targets.")

            # Query for Sales Data
            sales_query = """
            SELECT fs.order_date AS InvoiceDate, SUM(fs.sales) AS TotalPrice
            FROM public.fact_sales fs
            GROUP BY fs.order_date
            ORDER BY fs.order_date;
            """
            sales_data = load_data_from_db(sales_query)

            # Process sales data for forecasting
            sales_data.columns = sales_data.columns.str.lower()
            sales_data["invoicedate"] = pd.to_datetime(sales_data["invoicedate"])
            monthly_sales = sales_data.groupby(sales_data["invoicedate"].dt.to_period("M"))["totalprice"].sum().reset_index()
            monthly_sales.columns = ["Month", "Sales"]
            monthly_sales["MonthIndex"] = range(1, len(monthly_sales) + 1)

            # Train/Test Split
            X = monthly_sales[["MonthIndex"]]
            y = monthly_sales["Sales"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.write(f"Mean Squared Error: {mse:.2f}")

            # Visualization: Actual vs Predicted Sales
            st.write("This scatter plot shows the actual vs. predicted sales using the linear regression model. It helps to evaluate the accuracy of the model.")
            fig_actual_pred = go.Figure()
            fig_actual_pred.add_trace(go.Scatter(x=X_test["MonthIndex"], y=y_test, mode="markers", name="Actual"))
            fig_actual_pred.add_trace(go.Scatter(x=X_test["MonthIndex"], y=y_pred, mode="lines", name="Predicted"))
            fig_actual_pred.update_layout(title="Actual vs Predicted Sales", xaxis_title="Month Index", yaxis_title="Sales")
            st.plotly_chart(fig_actual_pred)

            # Future Sales Forecast
            st.write("This line chart forecasts future sales based on the linear regression model. It helps in planning for upcoming months.")
            future_months = pd.DataFrame({"MonthIndex": range(len(monthly_sales) + 1, len(monthly_sales) + 13)})
            future_sales = model.predict(future_months)
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=monthly_sales["MonthIndex"], y=monthly_sales["Sales"], mode="lines", name="Historical Sales"))
            fig_forecast.add_trace(go.Scatter(x=future_months["MonthIndex"], y=future_sales, mode="lines", name="Forecasted Sales"))
            fig_forecast.update_layout(title="Future Sales Forecast", xaxis_title="Month Index", yaxis_title="Sales")
            st.plotly_chart(fig_forecast)
