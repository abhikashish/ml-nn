import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r"/workspaces/ml-nn/datasets/auto-mpg.csv")

# Replace missing values represented by '?'
data.replace('?', pd.NA, inplace=True)

# Convert horsepower to numeric
data['horsepower'] = pd.to_numeric(data['horsepower'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['horsepower']] = imputer.fit_transform(data[['horsepower']])

# Select input features
X = data[['cylinders', 'displacement', 'horsepower',
          'weight', 'acceleration', 'model-year']]

# Target variable
y = data['mpg']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict MPG
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Display Actual vs Predicted values
result = pd.DataFrame({
    'Actual MPG': y_test,
    'Predicted MPG': y_pred
})
print(result.head())