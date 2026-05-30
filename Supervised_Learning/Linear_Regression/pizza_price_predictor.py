# ------------------------------------------------------------------
# Goal: Train a model, read coefficients, and make a prediction.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dataset
data = pd.DataFrame({
    'size_inches': [10, 12, 14, 16, 10, 12, 14, 16],
    'toppings':    [1,  1,  2,  2,  3,  3,  4,  4],
    'price':       [8, 10, 12, 14, 11, 13, 15, 17]
})


# 1. Separate features and target
X = data[['size_inches','toppings']]
y = data['price']

# 2. Train a model
model = LinearRegression()
model.fit(X,y)

# Predict on training data
y_pred = model.predict(X)

# 3. Print what the model learned
print(f"Intercept (base price) : ${model.intercept_:.2f}")
print(f"Coefficient for size : {model.coef_[0]:.2f}")
print(f"Coefficient for toppings : {model.coef_[1]:.2f}")

# 4. Predict new pizza price
pizza_size = float(input("Enter Pizza size : "))
pizza_topping = int(input("Number of toppings : "))

new_pizza = pd.DataFrame({'size_inches': [pizza_size], 'toppings':[pizza_topping]})
predicted_price = model.predict(new_pizza)
print(f"\nPredicted price for {pizza_size} inches, {pizza_topping}-topping pizza is: ${predicted_price[0]:.2f}")



# ==========================================================
# VISUALIZATION 1: 3D Scatter + Regression Plane
# ==========================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual data points
ax.scatter(
    data['size_inches'],
    data['toppings'],
    data['price'],
    color='blue',
    s=120,
    marker='o',
    edgecolors='orange',
    label='Actual Data'
)

# Create regression surface
size_range = np.linspace(
    data['size_inches'].min(),
    data['size_inches'].max(),
    20
)

topping_range = np.linspace(
    data['toppings'].min(),
    data['toppings'].max(),
    20
)

size_grid, topping_grid = np.meshgrid(
    size_range,
    topping_range
)

price_grid = (
    model.intercept_
    + model.coef_[0] * size_grid
    + model.coef_[1] * topping_grid
)

# Regression plane
ax.plot_surface(
    size_grid,
    topping_grid,
    price_grid,
    alpha=0.4,
    cmap='viridis'
)

# User prediction point
ax.scatter(
    pizza_size,
    pizza_topping,
    predicted_price,
    color='red',
    s=250,
    marker='*',
    edgecolors='black',
    label='Your Pizza'
)

ax.set_title("Pizza Price Prediction Model", fontsize=14)
ax.set_xlabel("Pizza Size (inches)")
ax.set_ylabel("Number of Toppings")
ax.set_zlabel("Price ($)")
ax.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# VISUALIZATION 2: Actual vs Predicted
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    y_pred,
    s=120,
    color='green',
    edgecolors='black'
)

# Perfect prediction line
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--',
    linewidth=2,
    label='Perfect Prediction'
)

plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()