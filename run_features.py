import sys
import os
import pandas as pd
from sklearn import preprocessing
sys.path.append('./features')
import stft
import mfcc
import metadata

in_root = 'processed_data'
audio_features_path = 'features/audio_features.csv'
metadata_path = 'features/metadata.csv'
metadata_features_path = 'features/metadata_features.csv'
full_features_path = 'features/features.csv'
normalized_features_path = 'features/normalized_features.csv'

bad_patient_ids = ["HA001", "HA002"]

metadata_cols = []

# Generate audio features

def write_features(file, patient_dir, create_header):
    headers = []
    features = []

    stft_headers, stft_features = stft.get_features(patient_dir)
    headers = headers + stft_headers
    features = features + stft_features

    mfcc_headers, mfcc_features = mfcc_features.get_features(patient_dir)
    headers = headers + mfcc_headers
    features = features + mfcc_features

    if create_header:
        file.write(','.join(headers) + '\n')
    file.write(','.join(features) + '\n')

        # Iterate through all patient data
        for patient_id in patient_ids:
            try:
                features = get_features(patient_id)
                print features
                output_file.write('%s,' % patient_id)
                for feature in features[:-1]:
                    output_file.write('%f,' % feature)
                output_file.write('%f' % features[-1])
                output_file.write('\n')
            except:
                # do nothing
                print 'something'

def get_patient_ids(path, study):
    patient_ids = [file for file in os.listdir(audio_file_path) if file.startswith(study)]
    patient_ids = [patient_id for patient_id in patient_ids if patient_id not in bad_patient_ids]
    return patient_ids.sort()
    return ["%s/%s" % (path, patient_id) for patient_id in patient_ids]

patient_dirs = \
    + \
    get_patient_dirs('%s/ed' % in_root, 'ED') + \

file = open(audio_features_path, "w")
create_header=True
for patient_id in get_patient_ids('%s/healthy' % in_root, 'HA'):
    if create_header:
        file.write('id,')
    write_features(file, '%s/healthy/%s' % (in_root, patient_id), create_header)
    if create_header:
        create_header=False

for patient_id in get_patient_ids('%s/ed' % in_root, 'ED'):
    write_features(file, '%s/ed/%s' % (in_root, patient_id), create_header)

# Process metadata
sys.exit(0)
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
