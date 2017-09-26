python process_patient_data.py --patient_data patient_data.csv --output patient_data_processed.csv

python merge_csvs.py --in1 patient_data_processed.csv --in2 audio_features.csv --merge_on id --out features.csv
