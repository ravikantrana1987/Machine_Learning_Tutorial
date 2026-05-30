import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

# ---------------------------------------------------------
# Sample dataset with 4 features
# ---------------------------------------------------------

data = pd.DataFrame({
    'size_inches': [10,12,14,16,10,12,14,16,18,20],
    'toppings': [1,1,2,2,3,3,4,4,5,5],
    'crust_thickness': [1,1,2,2,1,2,2,3,3,3],
    'delivery_distance': [1,2,2,3,4,5,3,6,4,7],
    'price': [8,10,12,14,11,13,15,17,20,22]
})

# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

X = data[
    [
        'size_inches',
        'toppings',
        'crust_thickness',
        'delivery_distance'
    ]
]

y = data['price']

# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

model = LinearRegression()
model.fit(X, y)

# ---------------------------------------------------------
# Print coefficients
# ---------------------------------------------------------

print("Intercept:", model.intercept_)

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X)

print("\nR² Score:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# =========================================================
# VISUALIZATION 1
# Actual vs Predicted
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y,
    y_pred,
    s=120,
    color='dodgerblue',
    edgecolors='black'
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--',
    linewidth=2,
    label='Perfect Fit'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.grid(alpha=0.3)
plt.legend()

plt.show()

# =========================================================
# VISUALIZATION 2
# One plot per feature
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,10)
)

features = X.columns

for ax, feature in zip(axes.flatten(), features):

    ax.scatter(
        data[feature],
        y,
        color='blue',
        s=100,
        edgecolors='black'
    )

    ax.set_title(f"{feature} vs Price")
    ax.set_xlabel(feature)
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 3
# Partial Dependence Plots
# Shows model-fitted relationship
# =========================================================

fig, ax = plt.subplots(
    2,
    2,
    figsize=(12,10)
)

PartialDependenceDisplay.from_estimator(
    model,
    X,
    features=[0,1,2,3],
    ax=ax
)

plt.suptitle(
    "Partial Dependence Plots",
    fontsize=16
)

plt.tight_layout()
plt.show()

# =========================================================
# VISUALIZATION 4
# Coefficient Importance
# =========================================================

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coef_df = coef_df.sort_values(
    by='Coefficient',
    ascending=False
)

plt.figure(figsize=(8,5))

bars = plt.bar(
    coef_df['Feature'],
    coef_df['Coefficient'],
    color='skyblue',
    edgecolor='black'
)

plt.title("Feature Importance (Linear Regression)")
plt.ylabel("Coefficient Value")
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f'{bar.get_height():.2f}',
        ha='center',
        va='bottom'
    )

plt.show()