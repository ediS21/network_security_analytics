import pandas as pd

# Load the dataset (assuming CSV format)
df = pd.read_csv('balanced_data_with_encoded_labels.csv')

# Specify the columns to keep based on feature importance and labels
columns_to_keep = [
    'startOffset', 'start', 'endOffset', 'end', 'sAckDelayAvg', 'rMACs', 'rBytesSum',
    'rInterPacketAvg', 'rAckDelayAvg', 'sInterPacketAvg', 'rPackets', 'rPayloadSum',
    'rLoad', 'duration', 'sBytesSum', 'sPackets', 'rAddress', 'sPayloadSum', 'rAckDelayMax',
    'sAckDelayMax', 'rIPs', 'sLoad', 'rPayloadAvg', 'sMACs', 'rttl', 'rBytesAvg', 'sttl',
    'rPshRate', 'sBytesAvg', 'sPshRate', 'rPayloadMin', 'sPayloadAvg', 'sPayloadMin',
    'rBytesMin', 'rAckDelayMin', 'sIPs', 'sBytesMin', 'rWinTCP', 'sAddress',
    'IT_M_Label', 'NST_M_Label'
]

# Filter the dataset to include only the selected features and labels
df_filtered = df[columns_to_keep]

# Save the filtered dataset to a new CSV file
df_filtered.to_csv('ICS-Flow_filtered_features_39features.csv', index=False)

# Display the first few rows of the filtered dataset
print(df_filtered.head())
