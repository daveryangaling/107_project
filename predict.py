import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Connect to the PostgreSQL database
engine = create_engine('postgresql://postgres:admin@localhost:5432/is107')

# Load the sales data from the fact_sales table
sales_data = pd.read_sql('SELECT order_date, sales FROM fact_sales', engine)

# Convert date field to datetime format
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])

# Aggregate only the 'sales' column by month
monthly_sales = sales_data.groupby(sales_data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()

# Prepare data for regression
X = np.arange(len(monthly_sales)).reshape(-1, 1)  # Time as a single feature
y = monthly_sales['sales']  # Sales amount

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['order_date'], y, label='Actual Sales')
plt.plot(monthly_sales['order_date'][len(X_train):], y_pred, label='Predicted Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Monthly Sales')
plt.legend()
plt.show()
