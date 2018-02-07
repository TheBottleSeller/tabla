import argparse
import scipy.io.wavfile

import onsetdetection

import madmom
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

parser = argparse.ArgumentParser(description='Isolate the chirp in the PS files.')
parser.add_argument('--input', type=str, help='file path to audio file')
parser.add_argument('--output', type=str, help='file output file path')
args = parser.parse_args()

sample_rate = 4000
length = 15

# TODO: Add validation that this is a PS sound

# sr, audio = scipy.io.wavfile.read(args.input)
# print audio
# onsets = onsetdetection.detect_onsets(audio)
# print onsets
# print onsets[-2] * 1.0 / sample_rate
window_size = 256
num_frames = length * sample_rate / window_size
spec = madmom.audio.spectrogram.Spectrogram(args.input, num_frames=num_frames)
print spec
print spec.shape
sf = madmom.features.onsets.spectral_flux(spec)
# calculate the difference
diff = np.diff(spec, axis=0)
# keep only the positive differences
pos_diff = np.maximum(0, diff)
# sum everything to get the spectral flux
sf = np.sum(pos_diff, axis=1)
plt.figure()
plt.imshow(spec[:, :200].T, origin='lower', aspect='auto')
plt.figure()
plt.imshow(pos_diff[:, :200].T, origin='lower', aspect='auto')
plt.figure()
plt.plot(sf)
# print sf
# print sf.size
# log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(args.input, num_bands=24)
# superflux_3 = madmom.features.onsets.superflux(log_filt_spec)
# print superflux_3
# print superflux_3.size
print sf
print sf.shape
# print madmom.features.onsets.peak_picking(sf)

# import sys
# from aubio import source, onset
#
# win_s = 512                 # fft size
# hop_s = win_s // 2          # hop size
#
# filename = args.input
#
# samplerate = 0
#
# s = source(filename, samplerate, hop_s)
# samplerate = s.samplerate
#
# o = onset("complex", win_s, hop_s, samplerate)
#
# # list of onsets, in samples
# onsets = []
#
# # total number of frames read
# total_frames = 0
# while True:
#     samples, read = s()
#     if o(samples):
#         print("%f" % o.get_last_s())
#         onsets.append(o.get_last())
#     total_frames += read
#     if read < hop_s: break
# print len(onsets)
