import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt

## 1. Datasets
cars = pd.DataFrame({
    'horsepower': [100, 130, 150, 180, 200, 90, 120, 160],
    'weight_kg':  [1200, 1350, 1450, 1600, 1700, 1100, 1250, 1500],
    'cylinders':  [4, 6, 6, 8, 8, 4, 4, 6],
    'mpg':        [28, 24, 22, 18, 15, 30, 26, 20]
})

## 2. Separate features and targets
X = cars[['horsepower','weight_kg','cylinders']]
y = cars['mpg']

## 3. Train a Model
model = LinearRegression()
model.fit(X,y)

## 4. Predict on trained model
y_predict = model.predict(X)

## 5. Result analysis
print(f"Intercept (base price) : {model.intercept_:.2f}")
print(f"Coefficient of horsepower : {model.coef_[0]:.2f}")
print(f"Coefficient of weight : {model.coef_[1]:.02f}")
print(f"Coefficient of cylinders : {model.coef_[2]:.02f}")
# print(cars['horsepower']);

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature} : {coef:.3f}")


## 6. Evaluation
mae = mean_absolute_error(y,y_predict)
r2 = r2_score(y,y_predict)

print(f"MAE : {mae:.3f}")
print(f"r2 : {r2:.3f}")

