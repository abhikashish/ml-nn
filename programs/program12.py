import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGD Optimizer
sgd_model = MLPClassifier( hidden_layer_sizes=(10,), activation='relu', solver='sgd',
            learning_rate_init=0.01, momentum=0.0, max_iter=3000, random_state=42)
sgd_model.fit(X_train, y_train)
sgd_pred = sgd_model.predict(X_test)
print("SGD Optimizer")
print("Accuracy:", accuracy_score(y_test, sgd_pred))

# SGD with Momentum
momentum_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd',
                 learning_rate_init=0.01, momentum=0.9, max_iter=3000, random_state=42 )
momentum_model.fit(X_train, y_train)
momentum_pred = momentum_model.predict(X_test)
print("\nSGD with Momentum")
print("Accuracy:", accuracy_score(y_test, momentum_pred))