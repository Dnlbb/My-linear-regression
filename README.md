# Mercedes-Benz USA Price Prediction Model

## Project Overview

This project is centered around the independent implementation of a linear regression model to predict the prices of Mercedes-Benz cars in the USA. The model is crafted from scratch using NumPy and Python, highlighting the principles of linear regression and gradient descent optimization. Additionally, this project compares the results of our custom model with scikit-learn's built-in LinearRegression model to validate our implementation's effectiveness.

## Data Preparation

The data includes details such as mileage, user ratings, review counts, and car prices. Here are the steps taken to prepare the data for modeling:

1. Load the data from a CSV file.
2. Clean and preprocess the data by removing non-numeric characters and converting strings to numerical values.
3. Filter out entries without pricing data and handle missing values to ensure quality inputs for modeling.

## Implementation Details

### Custom Linear Regression Implementation

The focus of this project is on the custom implementation of the linear regression algorithm. This process involves manually coding the calculation of the Mean Squared Error (MSE) and its gradient, as well as implementing the gradient descent algorithm to optimize the model parameters.

```python
import numpy as np
import pandas as pd

# Load and preprocess data
df = pd.read_csv('/path/to/usa_mercedes_benz_prices.csv')
df['Mileage'] = df['Mileage'].astype(str).str.replace(' mi.', '').str.replace(',', '').astype(float)
df['Review Count'] = df['Review Count'].str.replace(',', '').astype(float)
df = df.dropna(subset=['Rating'])
df = df[df['Price'] != 'Not Priced']
df['Price'] = df['Price'].str.replace('$', '').str.replace(',', '').astype(float)
```

# Prepare features and target
```python
features = df[['Mileage', 'Rating', 'Review Count']].values
target = df['Price'].values
```
# Custom linear regression functions
```python
def MseError_mat(X, w, y):
    y_pred = X @ w
    return np.sum((y - y_pred) ** 2) / len(y_pred)

def gr_MseError_mat(X, w, y):
    y_pred = X @ w
    return 2 / len(X) * X.T @ (y_pred - y)

weights = np.zeros(features.shape[1])
learning_rate = 0.000000001
eps = 0.0001
```
# Gradient descent for weight optimization
```python
for i in range(1000):
    cur_weights = weights
    weights -= learning_rate * gr_MseError_mat(features, cur_weights, target)
    if np.linalg.norm(cur_weights - weights, ord=2) <= eps:
        break
```
## This project demonstrates the robustness of a self-implemented linear regression model in predicting car prices, backed by a comparative analysis with a well-established machine learning library. The process highlights the educational value of building models from the ground up and understanding the underlying mechanics of machine learning algorithms.

