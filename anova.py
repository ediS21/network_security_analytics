import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

# Sample dataset based on your input (you can load your CSV file)
data = pd.read_csv('preprocessing_data/processed_westermo.csv')

# List of features (excluding the labels and any irrelevant columns like start_hour, start_day, etc.)
features = ['sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol', 'start', 'end', 'startOffset', 'endOffset', 
            'duration', 'sPackets', 'rPackets', 'sBytesSum', 'rBytesSum', 'sBytesMax', 'rBytesMax', 'sBytesMin', 'rBytesMin', 
            'sBytesAvg', 'rBytesAvg', 'sLoad', 'rLoad', 'sPayloadSum', 'rPayloadSum', 'sPayloadMax', 'rPayloadMax', 
            'sPayloadMin', 'rPayloadMin', 'sPayloadAvg', 'rPayloadAvg', 'sInterPacketAvg', 'rInterPacketAvg', 'sttl', 'rttl', 
            'sAckRate', 'rAckRate', 'sUrgRate', 'rUrgRate', 'sFinRate', 'rFinRate', 'sPshRate', 'rPshRate', 'sSynRate', 
            'rSynRate', 'sRstRate', 'rRstRate', 'sWinTCP', 'rWinTCP', 'sFragmentRate', 'rFragmentRate', 'sAckDelayMax', 
            'rAckDelayMax', 'sAckDelayMin', 'rAckDelayMin', 'sAckDelayAvg', 'rAckDelayAvg']

# Labels
labels_it = data['IT_M_Label']
labels_nst = data['NST_M_Label']

# Perform ANOVA for each feature and store the p-values
anova_results_it = {}
anova_results_nst = {}

for feature in features:
    # ANOVA test between the feature and IT_M_Label
    f_val_it, p_val_it = stats.f_oneway(data[feature], labels_it)
    anova_results_it[feature] = p_val_it

    # ANOVA test between the feature and NST_M_Label
    f_val_nst, p_val_nst = stats.f_oneway(data[feature], labels_nst)
    anova_results_nst[feature] = p_val_nst

# Sorting features by p-value for IT_M_Label
sorted_features_it = sorted(anova_results_it.items(), key=lambda x: x[1])

# Sorting features by p-value for NST_M_Label
sorted_features_nst = sorted(anova_results_nst.items(), key=lambda x: x[1])

# Display the top 10 significant features for both labels
# print("Top 10 significant features for IT_M_Label based on ANOVA test:")
# print(sorted_features_it[:10])

# print("\nTop 10 significant features for NST_M_Label based on ANOVA test:")
# print(sorted_features_nst[:10])

# Sample RandomForest with IT_M_Label
rf = RandomForestClassifier()
rf.fit(data[features], data['IT_M_Label'])

# Get feature importance
importances = rf.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print(feature_importance)
