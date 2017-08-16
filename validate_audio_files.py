import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Validate pnuemonia audio files.')
parser.add_argument('--audio_file_path', type=str, help='file path to audio files')
parser.add_argument('--study', type=str, help='ED or PNA')
args = parser.parse_args()

study = args.study
audio_file_path = args.audio_file_path
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

def is_valid_audio_filename(filename):
    if not filename.endswith('.wav'):
        return False
    filename = filename[:-4]
    parts = filename.split('_')

    if not len(parts) == 2:
        return False

    if not represents_int(parts[1]):
        return False

    if (parts[0] != 'LLL' and
        parts[0] != 'LML' and
        parts[0] != 'LUL' and
        parts[0] != 'RLL' and
        parts[0] != 'RML' and
        parts[0] != 'RUL'):
        return False

    return True

# BS, PS, or TF
def verify_type_dir(type_dir):
    audio_files = [file for file in os.listdir(type_dir) if file != 'Icon\r']

    for file in audio_files:
        if not is_valid_audio_filename(file):
            raise Exception('Bad audio file name: %s' % file)

    def verify_no_missing_trials(location):
        recordings = [f for f in audio_files if f.startswith(location)]
        if len(recordings) == 0:
            raise Exception('Missing %s trials for %s' % (location, type_dir))

        for i in range(len(recordings)):
            expected_trial = "%s_%d.wav" % (location, i + 1)
            if expected_trial not in audio_files:
                raise Exception('Missing trial %s for %s' % (expected_trial, type_dir))

    verify_no_missing_trials('LLL')
    verify_no_missing_trials('LML')
    verify_no_missing_trials('LUL')
    verify_no_missing_trials('RLL')
    verify_no_missing_trials('RML')
    verify_no_missing_trials('RUL')

def verify_patient_dir(patient_dir, last_seen_id):
    # Check not missing a patient data dir
    id = int(patient_dir[len(study):])
    if not id == last_seen_id + 1:
        print('Missing patient data %d' % (last_seen_id + 1))
        sys.exit(-1)

    # Check BS data
    bs_path = "%s%s/BS" % (audio_file_path, patient_dir)
    if os.path.isdir(bs_path):
        verify_type_dir(bs_path)

    # Check PS data
    ps_path = "%s%s/PS" % (audio_file_path, patient_dir)
    if os.path.isdir(ps_path):
        verify_type_dir(ps_path)

    # Check TF data
    tf_path = "%s%s/TF" % (audio_file_path, patient_dir)
    if os.path.isdir(tf_path):
        verify_type_dir(tf_path)

last_seen_id = 0
for patient_dir in patient_dirs:
    try:
        verify_patient_dir(patient_dir, last_seen_id)
    except Exception as error:
        print('Failed validation patient %s' % patient_dir)
        print(error)
    last_seen_id = last_seen_id + 1
