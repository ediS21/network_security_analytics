import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset, skipping rows with NaN values
data = pd.read_csv('balanced_data_with_encoded_labels.csv').dropna()

# Split the data into features and labels
X = data.iloc[:, 1:]  # Features (all columns except the first)
y = data.iloc[:, 0]   # Labels (first column)

# Split the dataset into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Combine labels and features back into DataFrames
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

# Save to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Data successfully split and saved into 'train_data.csv' and 'test_data.csv'.")
