import pandas as pd

df = pd.read_csv('data/ics-flow/Dataset.csv')

# Retrieve protocol with a count of values from the dataset
print(df['protocol'].value_counts())