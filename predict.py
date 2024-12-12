import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

# Display Main Title
st.title("Business Intelligence Project: Web Application Retail Store Forecasting and Analysis")

# Data Visualization Module
if app_mode == "Data Visualization":
    st.title("Data Visualization Module")

    query = """
    SELECT 
        fs.sales_id,
        fs.sales AS total_sales,
        fs.order_date,
        dc.customer_name,
        dl.region,
        dl.state,
        dl.city,
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

        # Calculate Key Metrics
        total_sales = data["total_sales"].sum()
        total_orders = data["sales_id"].nunique()
        unique_customers = data["customer_name"].nunique()

        # Display Key Metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Orders", f"{total_orders}")
        col3.metric("Unique Customers", f"{unique_customers}")

        # Filters
        st.sidebar.header("Filter Options")
        data["order_date"] = pd.to_datetime(data["order_date"])

        # Date Range Filter
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range",
            [data["order_date"].min().date(), data["order_date"].max().date()]
        )
        if start_date and end_date:
            data = data[(data["order_date"] >= pd.Timestamp(start_date)) & (data["order_date"] <= pd.Timestamp(end_date))]

        # Region, State, City Filters
        region = st.sidebar.selectbox("Select Region", ["All"] + list(data["region"].dropna().unique()))
        if region != "All":
            data = data[data["region"] == region]

        state = st.sidebar.selectbox("Select State", ["All"] + list(data["state"].dropna().unique()))
        if state != "All":
            data = data[data["state"] == state]

        city = st.sidebar.selectbox("Select City", ["All"] + list(data["city"].dropna().unique()))
        if city != "All":
            data = data[data["city"] == city]

        # Display Filter Summary
        st.markdown(f"""
        **Filter Summary**
        - Date Range: {start_date} to {end_date}
        - Region: {region}
        - State: {state}
        - City: {city}
        """)

        # Visualizations
        st.write("### Visualizations")

        # 1. Sales by Category
        st.write("#### Sales by Category")
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
        st.write("This bar chart shows the total sales for each product category. It helps to identify which categories contribute the most to total sales.")

        # 2. Top 10 Selling Products
        st.write("#### Top 10 Selling Products")
        top_10_products = data.groupby("product_name")["total_sales"].sum().reset_index().sort_values(by="total_sales", ascending=False).head(10)
        fig_top_products = px.bar(
            top_10_products,
            x="product_name",
            y="total_sales",
            title="Top 10 Selling Products",
            labels={"product_name": "Product Name", "total_sales": "Sales ($)"},
            color="total_sales",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_top_products, use_container_width=True)
        st.write("This bar chart displays the top 10 selling products based on total sales. It helps to pinpoint the most popular products in your sales data.")

        # 3. Sales by Region
        st.write("#### Sales by Region")
        sales_by_region = data.groupby("region")["total_sales"].sum().reset_index()
        fig2 = px.pie(
            sales_by_region,
            values="total_sales",
            names="region",
            title="Sales Distribution by Region"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.write("This pie chart shows the distribution of total sales across different regions. It helps to understand which region contributes the most to sales.")

        # 4. Sales by State (Pie chart)
        st.write("#### Sales by State")
        sales_by_state = data.groupby("state")["total_sales"].sum().reset_index()
        fig4_state = px.pie(
            sales_by_state,
            values="total_sales",
            names="state",
            title="Sales Distribution by State"
        )
        st.plotly_chart(fig4_state, use_container_width=True)
        st.write("This pie chart shows how total sales are distributed across various states. It helps to quickly compare sales performance between states.")

        # 5. Sales by City (Pie chart)
        st.write("#### Sales by City")
        sales_by_city = data.groupby("city")["total_sales"].sum().reset_index()
        fig4_city = px.pie(
            sales_by_city,
            values="total_sales",
            names="city",
            title="Sales Distribution by City"
        )
        st.plotly_chart(fig4_city, use_container_width=True)
        st.write("This pie chart shows the distribution of sales across cities. It highlights which cities contribute the most to overall sales.")

        # 6. Monthly Sales Trends
        st.write("#### Monthly Sales Trends")
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
        st.write("This line chart shows the monthly trend of total sales over time. It helps to identify seasonal sales patterns or growth trends.")

        # Download Filtered Data
        st.write("### Download Filtered Data")
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="filtered_sales_data.csv",
            mime="text/csv",
        )

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
            st.write("This section involves segmenting customers into different clusters based on their spending behavior, purchase frequency, and recency. It helps in identifying different customer groups for targeted marketing strategies.")
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
    fig_actual_pred.update_layout(
        title="Actual vs Predicted Sales",
        xaxis_title="Month Index",
        yaxis_title="Sales ($)",
        legend_title="Legend",
        template="plotly_white"
    )
    st.plotly_chart(fig_actual_pred)

    # Display the table for Actual vs Predicted Sales
    actual_pred_df = pd.DataFrame({
        'Month Index': X_test["MonthIndex"],
        'Actual Sales ($)': y_test,
        'Predicted Sales ($)': y_pred
    })
    st.write("#### Actual vs Predicted Sales Table")
    st.dataframe(actual_pred_df)

    # Visualization: Future Sales Forecast
    future_months = pd.DataFrame({"MonthIndex": range(len(monthly_sales) + 1, len(monthly_sales) + 13)})
    future_sales = model.predict(future_months)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=monthly_sales["MonthIndex"], y=monthly_sales["Sales"], mode="lines", name="Historical Sales"))
    fig_forecast.add_trace(go.Scatter(x=future_months["MonthIndex"], y=future_sales, mode="lines", name="Forecasted Sales"))
    fig_forecast.update_layout(title="Future Sales Forecast", xaxis_title="Month Index", yaxis_title="Sales")
    st.plotly_chart(fig_forecast)

    # Display the table for Future Predicted Sales
    future_sales_df = pd.DataFrame({
        'Month Index': future_months["MonthIndex"],
        'Forecasted Sales ($)': future_sales
    })
    st.write("#### Future Predicted Sales Table")
    st.dataframe(future_sales_df)

