# 📉 The Problem: When Data Curves 🌊
# Imagine tracking a plant's growth vs. how much water you give it:
# 💧 Too little water → plant stays small
# 💧💧 Just right → plant grows fast
# 💧💧💧💧💧 Too much → roots rot, plant shrinks
# If you plot this, you get a hill-shaped curve, not a straight line.
# 🧠 Analogy: Trying to fit a rigid ruler to a curved mountain road. No matter how you tilt it, 
# large sections will be far from the road. A straight line underfits curved data.
# 💡 The Trick: It's Still Linear Regression!

# y = w₀ + w₁·x + w₂·x² + w₃·x³
# 🔑 Key Insight: The model is still "linear" because it's linear in the weights (w₀, w₁, w₂...), 
# not in the input x. The math stays identical. We just give it smarter ingredients!
# 🛠️ Feature Engineering = Creating new, more useful features from existing ones. x² is your first engineered feature!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score

# Generate curved data: y ≈ 2 + 3x - 0.5x² + noise
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 + 3*x - 0.5*x**2 + np.random.normal(0, 1.5, 50)

print(x)
print(y)

# Put in a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Reshape for sklearn
X = df[['x']]
y = df['y']

# Fit linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predict & evaluate
y_pred_lin = lin_reg.predict(X)
print(f"Linear R²: {r2_score(y, y_pred_lin):.3f}")
print(f"Linear MAE: {mean_absolute_error(y, y_pred_lin):.2f}")

# Transform x → [x, x²]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original X shape:", X.shape)
print("Polynomial X shape:", X_poly.shape)  # Now 2 columns!
print("First 3 rows:\n", X_poly[:3])

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

y_pred_poly = poly_reg.predict(X_poly)
print(f"\nPolynomial R²: {r2_score(y, y_pred_poly):.3f}")
print(f"Polynomial MAE: {mean_absolute_error(y, y_pred_poly):.2f}")

plt.scatter(x, y, alpha=0.6, label='Actual Data')
plt.plot(x, y_pred_lin, color='red', label='Linear (Straight Line)')
plt.plot(x, y_pred_poly, color='green', linewidth=2, label='Polynomial (Curved)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()