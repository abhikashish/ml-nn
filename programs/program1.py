import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
data = pd.read_csv(r"/workspaces/ml-nn/datasets/diabetes.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
data[data.select_dtypes(include=['int64','float64']).columns] = imputer.fit_transform(
    data.select_dtypes(include=['int64','float64'])
)

# Encode categorical variables
encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])

# Normalize numerical data
scaler = MinMaxScaler()
num_cols = data.select_dtypes(include=['int64','float64']).columns
data[num_cols] = scaler.fit_transform(data[num_cols])

# Display summary statistics
print("Summary Statistics")
print(data.describe())

# Display first five records
print("\nPreprocessed Dataset")
print(data.head())