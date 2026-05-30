# Goal: Learn the real-world ML pipeline: split data → train → test → evaluate → predict on unseen data.
# 🛠️ Tasks
# Split data into 80% training and 20% testing
# Train model on training data only
# Evaluate on test data (never train on test!)
# Predict price for a brand new house
# Compare train vs test performance. Is the model overfitting?

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


## 1. DATASET
houses = pd.DataFrame({
    'size_sqft':  [1200, 1500, 1800, 1100, 2000, 1300, 1600, 1900, 1400, 1700],
    'bedrooms':   [2, 3, 3, 2, 4, 2, 3, 4, 3, 3],
    'age_years':  [5, 10, 15, 2, 20, 8, 12, 18, 7, 10],
    'dist_km':    [2, 5, 8, 1, 12, 3, 6, 10, 4, 7],
    'price_k':    [250, 280, 260, 220, 240, 265, 255, 230, 270, 250]
})

## 2. Extract Features and Target values
X = houses.drop('price_k',axis=1)
y = houses['price_k']

## 3. SPLIT the Train / Test Data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, 
    random_state=42
)

## 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

## 5. Get Predicted values for Trained / Test values
# 5.1 trained prediction
train_prediction = model.predict(X_train)

# 5.2 test prediction
test_prediction = model.predict(X_test)


## 6. Evaluate the model prediction.
train_mae = mean_absolute_error(y_train, train_prediction)
test_mae = mean_absolute_error(y_test, test_prediction)

train_r2 = r2_score(y_train, train_prediction)
test_r2 = r2_score(y_test, test_prediction)

## Print the Model evaluations
print(f"Intercept : {model.intercept_:.03f}")
print("Coefficients of features")

for feat, coef in zip(X.columns, model.coef_):
    print(f"{feat} : {coef:.03}")

print("Train  MAE : ", train_mae)
print("Test  MAE : ", test_mae)
print("Train R2 : ", train_r2)
print("Test  R2 : ", test_r2)

## 7. Testing with new house
new_house = pd.DataFrame({
     'size_sqft':  [1550],
    'bedrooms':   [3],
    'age_years':  [11],
    'dist_km':    [6]
})

predicted_price = model.predict(new_house)
print(f"New house predicted price : ${predicted_price[0]:.03f}")
