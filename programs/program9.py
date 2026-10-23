import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Simple 2D dataset
X = np.array([
    [2, 3],
    [3, 3],
    [4, 5],
    [5, 6],
    [1, 1],
    [2, 1],
    [3, 2],
    [1, 2]
])

# Class labels
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Create Perceptron model
model = Perceptron(max_iter=1000, random_state=42)

# Train the model
model.fit(X, y)

# Accuracy
print("Training Accuracy:", model.score(X, y))

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict over mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', s=100)
plt.title("Decision Boundary using Single-Layer Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()