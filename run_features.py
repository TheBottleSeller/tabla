import sys
import os
import pandas as pd
from sklearn import preprocessing
sys.path.append('./features')
import stft
import mfcc
import centroid
import metadata
sys.path.append('./essentia/src/python/essentia')

in_root = 'processed_data'
audio_features_path = 'features/audio_features.csv'
metadata_path = 'features/metadata.csv'
metadata_features_path = 'features/metadata_features.csv'
full_features_path = 'features/features.csv'
normalized_features_path = 'features/normalized_features.csv'

# ED002 and ED004 are pneumothorax
bad_patient_ids = ["HA001", "HA002", "ED002", "ED004"]

metadata_cols = []

# Generate audio features

def write_features(file, patient_id, patient_dir, create_header):
    headers = ['id']
    features = [patient_id]

    # stft_headers, stft_features = stft.get_features(patient_dir)
    # headers = headers + stft_headers
    # features = features + stft_features

    mfcc_headers, mfcc_features = mfcc.get_features(patient_dir)
    headers = headers + mfcc_headers
    features = features + mfcc_features

    centroid_headers, centroid_features = centroid.get_features(patient_dir)
    headers = headers + centroid_headers
    features = features + centroid_features

    if create_header:
        file.write(','.join(headers) + ',\r')
    file.write(','.join(map(str, features)) + ',\r')


def get_patient_ids(path, study):
    patient_ids = [file for file in os.listdir(path) if file.startswith(study)]
    patient_ids = [patient_id for patient_id in patient_ids if patient_id not in bad_patient_ids]
    patient_ids.sort()
    return patient_ids


file = open(audio_features_path, "w")
create_header=True
for patient_id in get_patient_ids('%s/healthy' % in_root, 'HA'):
    write_features(file, patient_id, '%s/healthy/%s' % (in_root, patient_id), create_header)
    if create_header:
        create_header=False

# for patient_id in get_patient_ids('%s/ed' % in_root, 'ED'):
#     write_features(file, patient_id, '%s/ed/%s' % (in_root, patient_id), create_header)

for patient_id in get_patient_ids('%s/pna' % in_root, 'PNA'):
    write_features(file, patient_id, '%s/pna/%s' % (in_root, patient_id), create_header)

# Process metadata
metadata.process_metadata(metadata_path, metadata_features_path)

adamMakeThisFalseIfFails = True

audio_features = pd.read_csv(audio_features_path)
audio_features.drop('id', axis=1, inplace=True)
audio_features.to_csv(full_features_path, index=False)

# if adamMakeThisFalseIfFails:
#     audio_features = pd.read_csv(audio_features_path)
#     audio_features.drop('id', axis=1, inplace=True)
#     audio_features.to_csv(full_features_path, index=False)
#     sys.exit(0)
#
# # Merge audio and metadata features
# audio_features = pd.read_csv(audio_features_path)
# metadata_features = pd.read_csv(metadata_features_path)
#
# all_features = metadata_features.merge(audio_features, on='id', how="inner")
# all_features.drop('id', axis=1, inplace=True)
# all_features.drop(all_features.columns[len(all_features.columns)-1], axis=1, inplace=True)
# all_features.to_csv(full_features_path, index=False)

# Normalize features
# min_max_scaler = preprocessing.MinMaxScaler()
# normalized_features = pd.DataFrame(min_max_scaler.fit_transform(all_features))
# normalized_features.to_csv(normalized_features_path, index=False, header=False)
