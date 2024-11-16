import pandas as pd

# Load your CSV file (replace 'train.csv' with your file path)
df = pd.read_csv('train.csv')

column = 'Postal Code'

# Check for null values in the 'Postal Code' column
null_values = df[column].isnull()

# Count the number of null values in the column
null_count = null_values.sum()

# Output the result
print(f"Number of null values in {column}: {null_count}")

# Specify which rows have null values
null_rows = df[null_values]

# Output the rows that have null values
print(f"Rows with null values in {column}:\n{null_rows}")

# rows with null values in Postal Code:
# 2236
# 5276
# 8800
# 9148
# 9149
# 9150
# 9388
# 9389
# 9390
# 9391
# 9743