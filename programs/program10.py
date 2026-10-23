import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Iris dataset
iris = load_iris()

# Features and Target
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Create MLP with one hidden layer (10 neurons)
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                      solver='adam', max_iter=3000, random_state=42 )

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))