import pandas as pd

# Load the dataset
df = pd.read_csv('westermo_cleaned_data_IPV4-TCP.csv')

# Get unique values and their counts for IT_M_Label
it_m_label_counts = df['IT_M_Label'].value_counts()
print("IT_M_Label unique values and counts:")
print(it_m_label_counts)

# Get unique values and their counts for NST_M_Label
nst_m_label_counts = df['NST_M_Label'].value_counts()
print("\nNST_M_Label unique values and counts:")
print(nst_m_label_counts)

# Your existing conditions
condition1 = (df['IT_M_Label'] == 'Normal') & (df['NST_M_Label'] == 'Normal')
condition2 = df['NST_M_Label'] == 'Normal'
condition3 = df['IT_M_Label'] == 'Normal'

# Count matching rows for condition1
num_matching_rows = condition1.sum()
print("\nNumber of rows where both IT_M_Label and NST_M_Label are 'Normal':")
print(num_matching_rows)
