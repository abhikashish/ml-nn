import numpy as np
from sklearn.linear_model import Perceptron

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Outputs for AND and OR gates
outputs = {"AND": np.array([0, 0, 0, 1]),"OR": np.array([0, 1, 1, 1])}

# Train and test the perceptron for each logic gate
for gate_name, y in outputs.items():
    print(f"{gate_name} Gate using Perceptron")
    # Create Perceptron model
    model = Perceptron(max_iter=1000, random_state=42)
    # Train the model
    model.fit(X, y)
    # Predict
    predictions = model.predict(X)
    # Display results
    print("\nInput\tExpected\tPredicted")
    for i in range(len(X)):
        print(f"{X[i]}\t{y[i]}\t\t{predictions[i]}")
    # Accuracy
    accuracy = model.score(X, y)
    print("\nAccuracy:", accuracy)
    # Display learned parameters
    print("Weights:", model.coef_)
    print("Bias:", model.intercept_)