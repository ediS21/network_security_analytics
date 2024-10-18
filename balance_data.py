import pandas as pd

df = pd.read_csv('ics-flow/preprocessed_dataset_IPV4-TCP.csv')

# Remove rows that are entirely empty (they mainly contained empty values such as ' ,, ')
df = df.dropna(how='any')

# save to csv
output_path = 'cleaned_data_IPV4-TCP.csv' 
df.to_csv(output_path, index=False)