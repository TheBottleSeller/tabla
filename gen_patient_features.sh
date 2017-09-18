python process_patient_data.py --patient_data patient_data.csv --output patient_data_processed.csv

python merge_csvs.py --in1 audio_features.csv --in2 patient_data_processed.csv --merge_on id --out features.csv
