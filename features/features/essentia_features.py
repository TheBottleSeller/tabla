import sys
sys.path.append('../essentia/src/python')
import essentia
import essentia.standard as es

# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
features, features_frames = es.MusicExtractor(
    lowlevelStats=['mean', 'stdev'],
    rhythmStats=['mean', 'stdev'],
    tonalStats=['mean', 'stdev'],
    analysisSampleRate=4000,
    lowlevelFrameSize=2048, 
    lowlevelHopSize=512,
    lowlevelWindowType='blackmanharris62'
)('../processed_data/ed/ED003/PS/PS_LLL_1.wav')

# See all feature names in the pool in a sorted order
print sorted(features.descriptorNames())
print faetures
