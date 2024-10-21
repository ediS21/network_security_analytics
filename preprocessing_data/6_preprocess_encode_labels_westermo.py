import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the CSV file
csv_file_path = os.path.join(current_dir, '../preprocessing_data/westermo_cleaned_data_IPV4-TCP.csv')

df_balanced = pd.read_csv(csv_file_path)

# map labels to values
label_mapping = {
    'Normal': 0,
    'BAD-SSH': 1,
    'BAD-MITM': 2,
    'BAD-MISCONF-DUPLICATION': 3,
    'BAD-MISCONF': 4,
    'BAD-PORTSCAN1': 5,
    'BAD-PORTSCAN2': 6,
    'GOOD-SSH': 7
}

# Replace the labels in both columns according to the mapping
df_balanced['IT_M_Label'] = df_balanced['IT_M_Label'].replace(label_mapping)
df_balanced['NST_M_Label'] = df_balanced['NST_M_Label'].replace(label_mapping)

df_balanced.to_csv('processed_westermo.csv', index=False)