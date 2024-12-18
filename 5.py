import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('advertising.csv')

# Display the first 10 rows
print(dataset.head(10))

# Check the shape of the dataset
print("Dataset Shape:", dataset.shape)

# Check for missing values
print("Missing Values:", dataset.isna().sum())

# Check for duplicate rows
print("Are there any duplicates?", dataset.duplicated().any())

# Boxplots
fig, axs = plt.subplots(3, figsize=(5, 15))
sns.boxplot(dataset['TV'], ax=axs[0])
sns.boxplot(dataset['Newspaper'], ax=axs[1])
sns.boxplot(dataset['Radio'], ax=axs[2])
plt.tight_layout()
plt.show()

# Distribution plot
sns.distplot(dataset['Sales'])
plt.show()

# Pairplot
sns.pairplot(dataset, vars=['TV', 'Newspaper', 'Radio', 'Sales'], height=4, aspect=1, kind='scatter')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Prepare data
X = dataset[['TV']]
Y = dataset['Sales']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Initialize the model
slr = LinearRegression()

# Train the model
slr.fit(X_train, y_train)

# Print intercept and coefficient
print(f'Intercept: {slr.intercept_}')
print(f'Coefficient: {slr.coef_}')

# Regression Equation
print(f'Regression Equation: Sales = {slr.intercept_} + {slr.coef_[0]} * TV')

# Plot the training data and regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, slr.intercept_ + slr.coef_[0] * X_train, color='red')
plt.show()

# Predictions
y_pred = slr.predict(X_test)

# Prediction results
print(f"Prediction for test set: {y_pred}")

# Predicted vs Actual values
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(slr_diff)

# R-squared value for the model
print(f'R-squared value: {slr.score(X, Y):.2f}')
