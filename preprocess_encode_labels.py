import pandas as pd

df_balanced = pd.read_csv('balanced_data_IPV4-TCP.csv')

# map labels to values
label_mapping = {
    'Normal': 0,
    'ip-scan': 1,
    'port-scan': 2,
    'replay': 3,
    'ddos': 4,
    'mitm': 5
}

# Replace the labels in both columns according to the mapping
df_balanced['IT_M_Label'] = df_balanced['IT_M_Label'].replace(label_mapping)
df_balanced['NST_M_Label'] = df_balanced['NST_M_Label'].replace(label_mapping)

df_balanced.to_csv('balanced_data_with_encoded_labels.csv', index=False)