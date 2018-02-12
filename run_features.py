import sys
sys.path.append('./features')
import features

in_root = 'processed_data'
output = 'features/audio_features.csv'

# Open file and create headers
file = open(output, "w")
file.write(','.join(features.get_features_headers()) + '\n')
features.process_audio_features('HA', '%s/healthy' % in_root, file)
features.process_audio_features('ED', '%s/ed' % in_root, file)
