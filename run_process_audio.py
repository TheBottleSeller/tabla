import sys
import os
import traceback
import logging
sys.path.append('./processed_data')
from process_audio import process_audio

in_root = 'raw_data'
out_root = 'processed_data'

failures = []

def walk_and_process_audio(path):
    in_dir = os.path.join(in_root, path)
    out_dir = os.path.join(out_root, path)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for file in os.listdir(in_dir):
        filepath = os.path.join(in_dir, file)
        if os.path.isdir(filepath):
            walk_and_process_audio(os.path.join(path, file))
        elif os.path.isfile(filepath) and filepath.endswith('.wav'):
            in_audio = filepath
            out_audio = os.path.join(out_dir, file)
            type = file[0:2]
            if not os.path.isfile(out_audio):
                try:
                    process_audio(type, in_audio, out_audio)
                    print '##########'
                except Exception as e:
                    print e
                    print '!!!!!!!!!!!!!'
                    print 'Failed processing: %s' % (in_audio)
                    print '!!!!!!!!!!!!!'
                    failures.append(in_audio)

walk_and_process_audio('')
print '!!!!!!!!!!!!!'
for f in failures:
    print 'Failed processing: %s' % (f)
print '!!!!!!!!!!!!!'
