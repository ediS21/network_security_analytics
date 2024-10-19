import pandas as pd

# Load the dataset (assuming CSV format)
df = pd.read_csv('balanced_data_with_encoded_labels.csv')

# Specify the columns to keep
columns_to_keep = [
    'startOffset', 'start', 'endOffset', 'IT_M_Label','NST_M_Label'
]

# Filter the dataset
df_filtered = df[columns_to_keep]

# Save or display the filtered dataset
df_filtered.to_csv('ICS-Flow_3features_IT_NST.csv', index=False)
print(df_filtered.head())
