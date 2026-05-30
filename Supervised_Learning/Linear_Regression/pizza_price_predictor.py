# ------------------------------------------------------------------
# Goal: Train a model, read coefficients, and make a prediction.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# 3. Print what the model learned
print(f"Intercept (base price) : ${model.intercept_:.2f}")
print(f"Coefficient for size : {model.coef_[0]:.2f}")
print(f"Coefficient for toppings : {model.coef_[1]:.2f}")

# 4. Predict new pizza price
pizza_size = float(input("Enter Pizza size : "))
pizza_topping = int(input("Number of toppings : "))

new_pizza = pd.DataFrame({'size_inches': [pizza_size], 'toppings':[pizza_topping]})
predicted_price = model.predict(new_pizza)
print(f"\nPredicted price for {pizza_size} inches, {pizza_topping}-topping pizza: {predicted_price[0]:.2f}")
