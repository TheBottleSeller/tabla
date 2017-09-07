import argparse
import os
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='Merge audio features and patient data.')
parser.add_argument('--audio_features', type=str, help='file path to patient audio features')
parser.add_argument('--patient_data', type=str, help='file path to patient data')
args = parser.parse_args()

audio_features_path = args.audio_features
patient_data_path = args.patient_data
if not audio_features_path or not patient_data_path:
    print('missing file paths')
    sys.exit(-1)

audio_features = pd.read_csv(audio_features_path)
patient_data = pd.read_csv(patient_data_path)
merged = patient_data.merge(audio_features, on="id", how="outer").fillna("")
merged.to_csv("patient_features.csv", index=False)
