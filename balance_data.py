import pandas as pd

# Remove 15k entries where both labels are Normal to balance the ICS-Flow dataset
df = pd.read_csv('ICS-Flow_filtered_features_39features.csv')

# Retrieve the indices of rows where both labels are 'Normal'
normal_indices = df[(df['IT_M_Label'] == 0) & (df['NST_M_Label'] == 0)].index

# Randomly select 10,000 indices to remove
indices_to_remove = normal_indices.to_series().sample(n=10000, random_state=1)

# Drop the randomly selected indices
df_balanced = df.drop(indices_to_remove)

# Reset the index
df_balanced.reset_index(drop=True, inplace=True)
df_balanced.to_csv('balanced_data_ICS-Flow_39feat.csv', index=False)