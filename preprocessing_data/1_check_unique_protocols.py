import os
import pandas as pd

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the CSV file
csv_file_path = os.path.join(current_dir, '../westermo/flows/westermo.csv')

df = pd.read_csv(csv_file_path)

# Retrieve protocol with a count of values from the dataset
print(df['protocol'].value_counts())