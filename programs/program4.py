# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv(r"/workspaces/ml-nn/datasets/glass.csv")

# Step 3: Display Dataset
print("First 10 Records")
print(df.head(10))
print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

# Step 4: Features and Target
X = df.drop("Type", axis=1)
y = df["Type"]

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Find Best K Value
accuracy = []
print("\nAccuracy for Different K Values")
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    print("K =", k, " Accuracy =", round(acc, 4))

# Step 8: Best K
best_k = accuracy.index(max(accuracy)) + 1
print("\nBest K Value =", best_k)

# Step 9: Final Model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 10: Evaluation
print("\nAccuracy Score")
print(accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Step 11: Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\nActual vs Predicted")
print(result.head(10))

# Step 12: Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,16), accuracy, marker='o')
plt.title("Effect of K Value on Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(range(1,16))
plt.grid(True)
plt.show()