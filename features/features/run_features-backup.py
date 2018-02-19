import sys
import os
import traceback
import logging
import features

in_root = 'processed_data'
output = 'features/audio_features.csv'

failures = []
bad_patient_ids = ["HA001", "HA002", "HA007", "HA008"]

def walk_and_process_features(path, out_file):
    in_dir = os.path.join(in_root, path)

    def is_patient_dir(dirname):
        return dirname.startswith('ED') or dirname.startswith('HA')

    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        if os.path.isdir(filepath):
            if is_patient_dir(file):
                patient_id = file
                if patient_id in bad_patient_ids:
                    continue
                # try:
                print 'Processing: %s' % patient_id
                f = features.process_patient_features(filepath, patient_id)
                out_file.write(','.join(f) + '\n')
                # except Exception as e:
                #     print e
                #     print '!!!!!!!!!!!!!'
                #     print 'Failed processing: %s' % (filepath)
                #     print '!!!!!!!!!!!!!'
                #     failures.append(filepath)
                #     raise e
            else:
                walk_and_process_features(os.path.join(path, file), out_file)

# Open file and create headers
file = open(output, "w")
file.write(','.join(features.get_features_headers()) + '\n')

walk_and_process_features('', file)

print '!!!!!!!!!!!!!'
for f in failures:
    print 'Failed processing: %s' % (f)
print '!!!!!!!!!!!!!'
