# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 2: Load Dataset
df = pd.read_csv(r"/workspaces/ml-nn/datasets/breast-cancer.csv")

# Step 3: Display Dataset
print("First 10 Records:")
print(df.head(10))

# Step 4: Define Features and Target
# Drop both target and diagnosis columns from features
X = df.drop(['id', 'diagnosis'], axis=1)

# Target Variable
y = df['diagnosis']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Create Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Step 8: Train Model
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 12: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Actual vs Predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted (First 10 records)")
print(results.head(10))