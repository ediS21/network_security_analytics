import pandas as pd

# Load the dataset
df = pd.read_csv('processed_westermo.csv')

# Specify the columns to keep based on provided features and labels
columns_to_keep = [
    'endOffset', 'end', 'startOffset', 'start', 'rInterPacketAvg', 'sInterPacketAvg', 
    'duration', 'rAckDelayAvg', 'sAckDelayAvg', 'sLoad', 'rLoad', 'rBytesSum', 
    'rBytesAvg', 'sAckDelayMax', 'rPayloadAvg', 'rAckDelayMax', 'rMACs', 'sMACs', 
    'rPayloadSum', 'sWinTCP', 'rAddress', 'sPshRate', 'rPshRate', 'sBytesSum', 
    'sPackets', 'sIPs', 'rPackets', 'rAckRate', 'sAckDelayMin', 'rPayloadMax', 
    'sAddress', 'rBytesMax', 'sPayloadSum', 'sPayloadAvg', 'rAckDelayMin', 
    'sBytesAvg', 'rIPs', 'rSynRate', 'sSynRate', 'sFinRate', 'IT_M_Label', 'NST_M_Label'
]

# Filter the dataset to include only the selected features and labels
df_filtered = df[columns_to_keep]

# Save the filtered dataset to a new CSV file
df_filtered.to_csv('westermo_filtered_features_39features.csv', index=False)

# Display the first few rows of the filtered dataset
print(df_filtered.head())
