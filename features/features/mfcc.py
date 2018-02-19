import os
import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 4000

# Calculate the mean mfcc from coefficients 1-12 for each PS recording
# Average all the mfcc results across the ps recordings
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
	mfcc = ess.MFCC(numberCoefficients = 12)
	x = ess.MonoLoader(filename=audio_path, sampleRate = fs)()
	mfccs = []

	for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):
	  mX = spectrum(window(frame))
	  mfcc_bands, mfcc_coeffs = mfcc(mX)
	  mfccs.append(mfcc_coeffs)
	mfccs = np.array(mfccs)

	headers = []
	features = []
	for i in range(0, 12):
		coefficients = mfccs[:,i]
		headers.append('mean_mfcc_%d' % i)
		features.append(np.mean(coefficients))

	# plt.figure(1, figsize=(9.5, 7))

	# plt.subplot(2,1,1)
	# plt.plot(np.arange(x.size)/float(fs), x, 'b')
	# plt.axis([0, x.size/float(fs), min(x), max(x)])
	# plt.ylabel('amplitude')
	# plt.title('x (speech-male.wav)')

	# plt.subplot(2,1,2)
	# numFrames = int(mfccs[:,0].size)
	# frmTime = H*np.arange(numFrames)/float(fs)
	# plt.pcolormesh(frmTime, 1+np.arange(12), np.transpose(mfccs[:,1:]))
	# plt.ylabel('coefficients')
	# plt.title('MFCCs')
	# plt.autoscale(tight=True)
	# plt.tight_layout()
	# plt.savefig('mfcc.png')
	# plt.show()

	return headers, features

# print get_features('../processed_data/ed/ED003')
# print get_features('../../sms-tools/workspace/Tabla_test/ED003_PS_LLL_1.wav')
