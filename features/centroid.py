import os
import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess
#import essentia.stats as esst

M = 1024
N = 1024
H = 512
fs = 4000

# Calculate the centroid for each PS recording
# Average all the centroid results across the ps recordings
def get_features(patient_dir):
	ps_recordings = [file for file in os.listdir("%s/PS" % patient_dir) if file.endswith(".wav")]
	ps_features = []
	headers = []
	for recording in ps_recordings:
		hs, features = _get_features("%s/PS/%s" % (patient_dir, recording))
		headers = hs
		ps_features.append(features)
	average_ps_features = np.mean(ps_features, axis=0)
	return headers, average_ps_features.tolist()

def _get_features(audio_path):
	spectrum = ess.Spectrum(size=N)
	window = ess.Windowing(size=M, type='hann')
	centroid = ess.Centroid(range=1)
	x = ess.MonoLoader(filename=audio_path, sampleRate = fs)()
	spectrumcentroid = []

	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):          
	  mX = spectrum(window(frame))
	  centroidvalues = centroid(mX)
	  spectrumcentroid.append(centroidvalues)            
	spectrumcentroid = np.array(spectrumcentroid)

	headers = ['mean centroid']
	features = [np.mean(spectrumcentroid)]
	#[np.mean(centroid)]

	#plt.figure(1, figsize=(9.5, 7))

	# plt.subplot(2,1,1)
	# plt.plot(np.arange(x.size)/float(fs), x, 'b')
	# plt.axis([0, x.size/float(fs), min(x), max(x)])
	# plt.ylabel('amplitude')
	# plt.title('x')

	# plt.subplot(2,1,2)
	# plt.plot(spectrumcentroid)
	# plt.ylabel('frequency (Hz)')
	# plt.title('time (sec)')
	# plt.autoscale(tight=True)
	# plt.tight_layout()
	# plt.savefig('centroid.png')
	# plt.show()

	return headers, features

# print get_features('../processed_data/ed/ED003/PS/PS_LLL_1.wav')
# print get_features('../../sms-tools/workspace/Tabla_test/ED003_PS_LLL_1.wav')
# print get_features('../../sms-tools/workspace/Tabla_test/sine500hz.wav')