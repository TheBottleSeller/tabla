import sys
import os
import pandas as pd
from sklearn import preprocessing
sys.path.append('./features')
import stft_features
import metadata

in_root = 'processed_data'
audio_features_path = 'features/audio_features.csv'
metadata_path = 'features/metadata.csv'
metadata_features_path = 'features/metadata_features.csv'
full_features_path = 'features/features.csv'
normalized_features_path = 'features/normalized_features.csv'

metadata_cols = []

# Generate audio features
file = open(audio_features_path, "w")
file.write(','.join(features.get_features_headers()) + '\n')
stft_features.process_audio_features('HA', '%s/healthy' % in_root, file)
stft_features.process_audio_features('ED', '%s/ed' % in_root, file)

# Process metadata
metadata.process_metadata(metadata_path, metadata_features_path)

# Merge audio and metadata features
audio_features = pd.read_csv(audio_features_path)
metadata_features = pd.read_csv(metadata_features_path)
all_features = metadata_features.merge(audio_features, on='id', how="inner")
all_features.drop('id', axis=1, inplace=True)
all_features.drop(all_features.columns[len(all_features.columns)-1], axis=1, inplace=True)
all_features.to_csv(full_features_path, index=False)

# Normalize features
min_max_scaler = preprocessing.MinMaxScaler()
normalized_features = pd.DataFrame(min_max_scaler.fit_transform(all_features))
normalized_features.to_csv(normalized_features_path, index=False, header=False)
