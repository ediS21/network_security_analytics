'''
Script to separate protocols of datasets into different files, one for each protocol, 
due to missing values in some columns for particular protocols
''' 

import pandas as pd

data = pd.read_csv('data/ics-flow/Dataset.csv')

# Get unique protocols
protocols = data['protocol'].unique()

# Create a new CSV file for each protocol
for protocol in protocols:
    protocol_data = data[data['protocol'] == protocol]
    output_file = f'{protocol}.csv'
    protocol_data.to_csv(output_file, index=False)

print("Separated protocols")