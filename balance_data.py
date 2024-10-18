import pandas as pd

# Remove 15k entries where both labels are Normal to balance the ICS-Flow dataset
df = pd.read_csv('cleaned_data_IPV4-TCP2.csv')

# Retrieve the indices of rows where both labels are 'Normal'
normal_indices = df[(df['IT_M_Label'] == 'Normal') & (df['NST_M_Label'] == 'Normal')].index

# Randomly select 15,000 indices to remove
indices_to_remove = normal_indices.to_series().sample(n=15000, random_state=1)

# Drop the randomly selected indices
df_balanced = df.drop(indices_to_remove)

# Reset the index
df_balanced.reset_index(drop=True, inplace=True)
df_balanced.to_csv('balanced_data_IPV4-TCP.csv', index=False)