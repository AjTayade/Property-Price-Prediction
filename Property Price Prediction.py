# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'housing_data.csv' with the actual file path)
df = pd.read_csv("C:\\Users\\ajayt\\Downloads\\Data_file - data_file.csv")

# Exploratory Data Analysis (EDA)
# Basic Info
print("Data Information:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Visualize distributions of numerical features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns].hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Visualize correlations among numerical features
corr_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Data Preprocessing
# Handle missing values (for simplicity, filling missing values with the median)
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df['households'] = df['households'].fillna(df['households'].median())

# Encoding Categorical Variables (Ocean proximity is categorical)
df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes

# Splitting data into features and target
X = df.drop('median_house_value', axis=1)  # Features
y = df['median_house_value']  # Target variable

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but usually important for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Development

# 1. Simple Linear Regression (Example using 'longitude' as a feature)
X_simple = X_train[['longitude']]  # Using 'longitude' as a simple example
X_test_simple = X_test[['longitude']]

simple_model = LinearRegression()
simple_model.fit(X_simple, y_train)

# Predict and evaluate
y_pred_simple = simple_model.predict(X_test_simple)
print("\nSimple Linear Regression Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_simple))
print("R^2 Score:", r2_score(y_test, y_pred_simple))

# 2. Multiple Linear Regression (using all features)
multiple_model = LinearRegression()
multiple_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_multiple = multiple_model.predict(X_test_scaled)
print("\nMultiple Linear Regression Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_multiple))
print("R^2 Score:", r2_score(y_test, y_pred_multiple))

# Visualizing predictions vs true values for Multiple Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_multiple, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions (Multiple Linear Regression)")
plt.show()

# Final Insights (Model Evaluation)
print("\nModel Performance (Multiple Linear Regression):")
print(f"R-squared: {r2_score(y_test, y_pred_multiple)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_multiple)}")

