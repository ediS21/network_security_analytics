'''
Script to separate protocols of datasets into different files, one for each protocol, 
due to missing values in some columns for particular protocols
''' 

import os
import pandas as pd

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the CSV file
csv_file_path = os.path.join(current_dir, '../westermo/flows/westermo.csv')

data = pd.read_csv(csv_file_path)

# Get unique protocols
protocols = data['protocol'].unique()

# Create a new CSV file for each protocol
for protocol in protocols:
    protocol_data = data[data['protocol'] == protocol]
    output_file = f'{protocol}.csv'
    protocol_data.to_csv(output_file, index=False)
