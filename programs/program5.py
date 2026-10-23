# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv(r"/workspaces/ml-nn/datasets/bank-full.csv")

# Display first 10 rows
print("First 10 Records:")
print(data.head(10))

# Encode Categorical Variables
encoder = LabelEncoder()
for column in data.select_dtypes(include=['object', 'string']).columns:
    data[column] = encoder.fit_transform(data[column])

# Separate Features and Target
X = data.drop("Target", axis=1)
y = data["Target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree Model
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["No","Yes"], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Classifier")
plt.show()