import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the CSV file
csv_file_path = os.path.join(current_dir, '../westermo/flows/3_preprocessed_dataset_IPV4-TCP.csv')


df = pd.read_csv(csv_file_path)

# Remove rows that are entirely empty (they mainly contained empty values such as ' ,, ')
df = df.dropna(how='any')

# save to csv
output_path = 'TEST_westermo_cleaned_data_IPV4-TCP.csv' 
df.to_csv(output_path, index=False)