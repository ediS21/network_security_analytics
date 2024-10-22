import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset, skipping rows with NaN values
data = pd.read_csv('test_ICS-Flow_39feats.csv').dropna()

# Split for IT_M_Label
X_it = data.drop(columns=["IT_M_Label"])  # Features for IT_M_Label
y_it = data["IT_M_Label"]  # Labels for IT_M_Label

# Split the data into train (60%) and temp (40% which will be split into dev and test)
X_it_train, X_it_temp, y_it_train, y_it_temp = train_test_split(X_it, y_it, test_size=0.4, random_state=42, stratify=y_it)

# Further split temp data into dev (50% of temp, i.e., 20% of total) and test (50% of temp, i.e., 20% of total)
X_it_dev, X_it_test, y_it_dev, y_it_test = train_test_split(X_it_temp, y_it_temp, test_size=0.5, random_state=42, stratify=y_it_temp)

# Combine IT labels and features back into DataFrames
it_train_data = pd.concat([y_it_train, X_it_train], axis=1)
it_dev_data = pd.concat([y_it_dev, X_it_dev], axis=1)
it_test_data = pd.concat([y_it_test, X_it_test], axis=1)

# Save IT splits to CSV
it_train_data.to_csv('train_data_it.csv', index=False)
it_dev_data.to_csv('dev_data_it.csv', index=False)
it_test_data.to_csv('test_data_it.csv', index=False)
