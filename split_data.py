import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset, skipping rows with NaN values
data = pd.read_csv('balanced_data_with_encoded_labels.csv').dropna()

# First split for NST_M_Label
X_nst = data.drop(columns=["NST_M_Label"])  # Features for NST_M_Label
y_nst = data["NST_M_Label"]  # Labels for NST_M_Label

X_nst_train, X_nst_test, y_nst_train, y_nst_test = train_test_split(X_nst, y_nst, test_size=0.25, random_state=42, stratify=y_nst)

# Combine NST labels and features back into DataFrames
nst_train_data = pd.concat([y_nst_train, X_nst_train], axis=1)
nst_test_data = pd.concat([y_nst_test, X_nst_test], axis=1)

# Save NST splits to CSV
nst_train_data.to_csv('train_data_nst.csv', index=False)
nst_test_data.to_csv('test_data_nst.csv', index=False)

# Second split for IT_M_Label
X_it = data.drop(columns=["IT_M_Label"])  # Features for IT_M_Label
y_it = data["IT_M_Label"]  # Labels for IT_M_Label

X_it_train, X_it_test, y_it_train, y_it_test = train_test_split(X_it, y_it, test_size=0.25, random_state=42, stratify=y_it)

# Combine IT labels and features back into DataFrames
it_train_data = pd.concat([y_it_train, X_it_train], axis=1)
it_test_data = pd.concat([y_it_test, X_it_test], axis=1)

# Save IT splits to CSV
it_train_data.to_csv('train_data_it.csv', index=False)
it_test_data.to_csv('test_data_it.csv', index=False)
