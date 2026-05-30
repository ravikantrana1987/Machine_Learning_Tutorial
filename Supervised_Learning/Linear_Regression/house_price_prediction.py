# ------------------------------------------------------------
# House Price Prediction using Linear Regression
# ------------------------------------------------------------

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Training Dataset
# ------------------------------------------------------------
# sizes  -> House size (in 100s of square feet)
# prices -> House price (in $1000s)
#
# Example:
# size = 1  -> 100 sq ft
# price = 150 -> $150,000
# ------------------------------------------------------------

sizes = [1, 2, 3, 4, 5]
# prices = [150, 200, 250, 300, 350]
prices = [145, 210, 240, 310, 360]


# ------------------------------------------------------------
# Prepare the dataset for scikit-learn
# ------------------------------------------------------------

# In machine learning:
#
# X -> Input features
# y -> Output/target values
#
# scikit-learn expects:
#
# X shape = (number_of_samples, number_of_features)
# y shape = (number_of_samples,)
#
# Since we currently have only ONE feature (house size),
# NumPy creates a 1D array like this:
#
# [1, 2, 3, 4, 5]
#
# But LinearRegression expects X to be a 2D table:
#
# [
#   [1],
#   [2],
#   [3],
#   [4],
#   [5]
# ]
#
# Therefore we reshape the data.

# reshape(-1, 1) means:
#
# -1 -> NumPy automatically calculates the number of rows
#  1 -> Create exactly 1 column (because we have 1 feature)

X = np.array(sizes).reshape(-1, 1)

# y is the target/output variable.
# For single-output regression, y can remain a 1D array.
y = np.array(prices)


# Create and Train a model
model = LinearRegression()
model.fit(X,y) # ← This is where gradient descent happens internally!

# Checked what we have learned
# If Coefficients are in negative means, if we increase the feature magnitute by 1,
# it will decrease the predicted value by (coefficient * 1)
# Example if coef is -1.2, then the house size increased by 1sqft then its value will be decreased by $-1.2
print(f"Slope (a) : {model.coef_[0]:.2f}")
print(f"intersept (b) : {model.intercept_:.2f}")
print(f"Model Score : {model.score(X,y)}")


# Predict New House Price
new_house = np.array([3.4]).reshape(-1,1)
predicted_price = model.predict(new_house)
print(f"House size {new_house} (in 100sqf) - Predicted price : ${predicted_price[0]:.2f}")

# -----------------------------
# Visalization
# -----------------------------
# Plot the data and the learned line
plt.scatter(sizes, prices,color='blue', label='Actual Data')
plt.plot(sizes, model.predict(X), color='red', label='Learned Line')

# Plot predicted house price
plt.scatter(
    new_house,
    predicted_price,
    color='green',
    s=120,
    marker='X',
    label='Predicted Price'
)

# Display coordinates of predicted point
plt.text(
    new_house[0][0],
    predicted_price[0],
    f'(  {new_house[0][0]}, {predicted_price[0]:.1f})',
    fontsize=9,
    color='green'
)

plt.xlabel('Size (100 of sq ft)')
plt.ylabel('Price ($1000s)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()