import argparse
import os
import sys
import pandas as pd
import numpy as np

def process_metadata(patient_data_path, output_path):
    if not patient_data_path:
        print('missing patient data file paths')
        sys.exit(-1)

    patient_data = pd.read_csv(patient_data_path).as_matrix()

    DEFAULT_SMOKING_PACKS = 0
    DEFAULT_TEMP = 98.6
    DEFAULT_BP_SYSTOLIC = 130
    DEFAULT_BP_DIASTOLIC = 85
    DEFAULT_HR = 90
    DEFAULT_RR = 18

    KNOWN_BAD_IDS = [
        'PNA001',
        'PNA002',
        'PNA003',
        'PNA004',
        'PNA005',
        'PNA006',
        'PNA007',
        'ED008',
    ]

    def default_thorax_circ(gender):
        if gender == 'M':
            return 80
        else:
            return 80

    def to_float(s):
        try:
            return float(s)
        except ValueError:
            return False

    def process_row(index):
        row_data = patient_data[index, :]
        processed_row = np.array([])

        id = row_data[0]
        if id in KNOWN_BAD_IDS:
            return processed_row

        processed_row = np.append(processed_row, id)

        age = row_data[2]
        processed_row = np.append(processed_row, age)

        gender = row_data[3]
        if gender != 'M' and gender != 'F':
            print 'Row %d: Bad gender: %s' % (index, gender)
            return np.array([])
        processed_row = np.append(processed_row, 1 if gender == 'M' else 0)
        processed_row = np.append(processed_row, 1 if gender == 'F' else 0)

        thorax_circ = to_float(row_data[7])
        if not isinstance(thorax_circ, float):
            print "USING DEFAULT thorax_circ"
            print thorax_circ
            print type(thorax_circ)
            thorax_circ = default_thorax_circ(gender)
        processed_row = np.append(processed_row, thorax_circ)

        smoking_packs = to_float(row_data[8])
        if not isinstance(smoking_packs, float):
            smoking_packs = DEFAULT_SMOKING_PACKS
        processed_row = np.append(processed_row, smoking_packs)

        temp = to_float(row_data[9])
        if not isinstance(temp, float):
            temp = DEFAULT_TEMP
        processed_row = np.append(processed_row, temp)

        bp = row_data[10] # 120/80
        parts = bp.split('/')
        if len(parts) == 2:
            bp_systolic = to_float(parts[0])
            bp_diastolic = to_float(parts[1])
            if not isinstance(bp_systolic, float) and isinstance(bp_diastolic, float):
                bp_systolic = DEFAULT_BP_SYSTOLIC
                bp_diastolic = DEFAULT_BP_DIASTOLIC
        else:
            bp_systolic = DEFAULT_BP_SYSTOLIC
            bp_diastolic = DEFAULT_BP_DIASTOLIC
        processed_row = np.append(processed_row, bp_systolic)
        processed_row = np.append(processed_row, bp_diastolic)

        hr = to_float(row_data[11])
        if not isinstance(hr, float):
            hr = DEFAULT_HR
        processed_row = np.append(processed_row, hr)

        rr = to_float(row_data[12])
        if not isinstance(rr, float):
            rr = DEFAULT_RR
        processed_row = np.append(processed_row, rr)

        spo02 = to_float(row_data[13])
        processed_row = np.append(processed_row, spo02)

        # sp02_at_recording = row_data[14]

        peak_flow = to_float(row_data[15])
        processed_row = np.append(processed_row, peak_flow)

        # Dangerous to include this because there are no healthy people that
        # wheeze or have shortness of breath
        presenting_symptom = row_data[19]
        sob = 0
        wheezing = 0
        if presenting_symptom == 'SOB and Wheezing':
            sob = 1
            wheezing = 1
        elif presenting_symptom == 'SOB':
            sob = 1
        elif presenting_symptom == 'Wheezing':
            wheezing = 1
        processed_row = np.append(processed_row, sob)
        processed_row = np.append(processed_row, wheezing)

        #### PNEUMONIA OUTPUT ###
        lung_disease = 1
        diagnosis = row_data[20]
        if diagnosis == 'Healthy' or diagnosis == 'None':
            lung_disease = 0
        processed_row = np.append(processed_row, lung_disease)

        return processed_row

    [rows, cols] = patient_data.shape

    headers = ['id', 'age', 'male', 'female', 'thorax_circ', 'smoking_packs', 'temp', 'bp_systolic', 'bp_diastolic', 'hr', 'rr', 'sp02', 'peak_flow', 'sob', 'wheezing', 'lung_disease']

    output_data = np.array([])
    for row_index in range(2, rows):
        next_row = process_row(row_index)
        if next_row.size != 0:
            if output_data.size == 0:
                output_data = next_row
            else:
                output_data = np.vstack([output_data, next_row])
    df = pd.DataFrame(data=output_data, columns=headers)
    df.to_csv(output_path, index=False, line_terminator='\n')
