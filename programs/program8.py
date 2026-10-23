import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Input data (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR Output
y = np.array([0, 1, 1, 0])

# Create Perceptron model
model = Perceptron(max_iter=1000, random_state=42)

# Train the model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Display results
print("Input\tExpected\tPredicted")
for i in range(len(X)):
    print(f"{X[i]}\t{y[i]}\t\t{y_pred[i]}")

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("\nAccuracy:", accuracy)

# Display learned parameters
print("Weights:", model.coef_)
print("Bias:", model.intercept_)