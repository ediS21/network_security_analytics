import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the CSV file
csv_file_path = os.path.join(current_dir, '../westermo/flows/preprocessed/IPV4-TCP.csv')

df = pd.read_csv(csv_file_path)

if not os.path.exists('mappings_IPV4-TCP'):
    os.makedirs('mappings_IPV4-TCP')

# label encode the categorical columns
label_columns = ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol']
label_encoders = {}

for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

    # Save the mapping (original value -> encoded value) to a CSV file
    mapping_df = pd.DataFrame({
        'original_value': le.classes_,
        'encoded_value': range(len(le.classes_))
    })
    mapping_df.to_csv(f'mappings_IPV4-TCP/{col}_mapping.csv', index=False)

# convert time-series data (startDate, endDate) to numerical values
def convert_datetime_features(df, start_col, end_col):
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    df['start_hour'] = df[start_col].dt.hour
    df['start_day'] = df[start_col].dt.day
    df['start_month'] = df[start_col].dt.month

    df['end_hour'] = df[end_col].dt.hour
    df['end_day'] = df[end_col].dt.day
    df['end_month'] = df[end_col].dt.month

    df['duration_seconds'] = (df[end_col] - df[start_col]).dt.total_seconds()

    # drop startDate and endDate columns
    df = df.drop(columns=[start_col, end_col])
    
    return df

df = convert_datetime_features(df, 'startDate', 'endDate')

# normalize the numeric columns using MinMaxScaler (good for LSTM)
scaler = MinMaxScaler()

# list of all numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# scale the numeric columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# save preprocessed data to new CSV file
df.to_csv('preprocessed_dataset_IPV4-TCP.csv', index=False)
