import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Ignore convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hidden layer sizes (model complexities)
hidden_sizes = [2, 4, 6, 8, 10, 15, 20, 30]
train_errors = []
test_errors = []

# Train MLP for different complexities
for size in hidden_sizes:
    model = MLPClassifier(hidden_layer_sizes=(size,), activation='relu', solver='adam',
                          max_iter=3000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Error = 1 - Accuracy
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)
    train_errors.append(train_error)
    test_errors.append(test_error)

# Print Results
print("Hidden Neurons\tTraining Error\tTesting Error")
for i in range(len(hidden_sizes)):
    print(f"{hidden_sizes[i]}\t\t{train_errors[i]:.4f}\t\t{test_errors[i]:.4f}")

# Plot Errors
plt.figure(figsize=(8,5))
plt.plot(hidden_sizes, train_errors, marker='o', linewidth=2, label='Training Error')
plt.plot(hidden_sizes, test_errors, marker='s', linewidth=2, label='Testing Error')
plt.title("Training vs Testing Error for Different Model Complexities")
plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
plt.show()