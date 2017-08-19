import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Validate pnuemonia audio files.')
parser.add_argument('--audio_file_path', type=str, help='file path to audio files')
parser.add_argument('--study', type=str, help='ED or PNA')
args = parser.parse_args()

study = args.study
audio_file_path = args.audio_file_path.rstrip('/')
if not audio_file_path:
    print('audio file path is required')
    sys.exit(-1)

if not study == 'PNA' and not study == 'ED':
    print('study must be either PNA or ED')
    sys.exit(-1)

patient_dirs = [file for file in os.listdir(audio_file_path) if file.startswith(study)]
patient_dirs.sort()

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_valid_audio_filename(type, filename):
    if not filename.endswith('.wav'):
        return False
    filename = filename[:-4]
    parts = filename.split('_')

    if not len(parts) == 3:
        print 'here1'
        return False
        
    if not parts[0] == type:
        print type
        return False
    
    if (parts[1] != 'LLL' and
        parts[1] != 'LML' and
        parts[1] != 'LUL' and
        parts[1] != 'RLL' and
        parts[1] != 'RML' and
        parts[1] != 'RUL'):
        return False

    if not represents_int(parts[2]):
        return False

    return True

# BS, PS, or TF
def verify_type_dir(patient, type):
    type_dir = "%s/%s/%s" % (audio_file_path, patient, type)
    if not os.path.isdir(type_dir):
        return
    
    audio_files = [file for file in os.listdir(type_dir) if file != 'Icon\r']

    for file in audio_files:
        if not is_valid_audio_filename(type, file):
            raise Exception('Bad audio file name: %s' % file)

    def verify_no_missing_trials(location):
        recordings = [f for f in audio_files if f.startswith("%s_%s" % (type, location))]
        if len(recordings) == 0:
            raise Exception('Missing %s trials for %s' % (location, type_dir))

        for i in range(len(recordings)):
            expected_trial = "%s_%s_%d.wav" % (type, location, i + 1)
            if expected_trial not in audio_files:
                raise Exception('Missing trial %s for %s' % (expected_trial, type_dir))

    verify_no_missing_trials('LLL')
    verify_no_missing_trials('LML')
    verify_no_missing_trials('LUL')
    verify_no_missing_trials('RLL')
    verify_no_missing_trials('RML')
    verify_no_missing_trials('RUL')

# patient PNA001, ED001
def verify_patient_dir(patient, last_seen_id):
    # Check not missing a patient data dir
    id = int(patient[len(study):])
    if not id == last_seen_id + 1:
        print('Missing patient data %d' % (last_seen_id + 1))
        sys.exit(-1)

    verify_type_dir(patient, "BS")
    verify_type_dir(patient, "PS")
    verify_type_dir(patient, "TF")

print "Validating %s study" % study

last_seen_id = 0
for patient_dir in patient_dirs:
    try:
        verify_patient_dir(patient_dir, last_seen_id)
    except Exception as error:
        print('Failed validation patient %s' % patient_dir)
        print(error)
    last_seen_id = last_seen_id + 1
    
print "Done"
