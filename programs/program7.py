import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 200)

# Step Function
def step(x):
    return np.where(x >= 0, 1, 0)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh Function
def tanh(x):
    return np.tanh(x)

# ReLU Function
def relu(x):
    return np.maximum(0, x)

# Plot all activation functions
plt.figure(figsize=(10, 8))

# Step Function
plt.subplot(2, 2, 1)
plt.plot(x, step(x), linewidth=2)
plt.title("Step Function")
plt.grid(True)

# Sigmoid Function
plt.subplot(2, 2, 2)
plt.plot(x, sigmoid(x), linewidth=2)
plt.title("Sigmoid Function")
plt.grid(True)

# Tanh Function
plt.subplot(2, 2, 3)
plt.plot(x, tanh(x), linewidth=2)
plt.title("Tanh Function")
plt.grid(True)

# ReLU Function
plt.subplot(2, 2, 4)
plt.plot(x, relu(x), linewidth=2)
plt.title("ReLU Function")
plt.grid(True)
plt.tight_layout()
plt.show()