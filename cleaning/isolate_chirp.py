import os
import sys
import numpy as np
from scipy.signal import get_window
import math
import matplotlib.pyplot as plt
import argparse
from pydub import AudioSegment
sys.path.insert(0, '../tools')
import stft
import utilFunctions as UF

window = 'blackman' # analysis window type (choice of rectangular, triangular, hanning, hamming, blackman, blackmanharris)
M = 961 # (integer): analysis window size (odd integer value)
N = 2048 # (integer): fft size (power of two, bigger or equal than than M)
H = 128 # (integer): hop size for the STFT computation

parser = argparse.ArgumentParser(description='Clean audio file')
parser.add_argument('--in', type=str, help='input file')
parser.add_argument('--out', type=str, help='output file')
input_path = args.in
output_path = args.out

def findChirpEnd(inputFile):
    def dB2energydB(mdB):
        m = 10 ** (mdB / 20.)
        energy_ = m ** 2.
        #m = 10 * np.log10(m.sum())
        energy_ = 10 * np.log10(np.sum(energy_))
        return energy_

    (fs, x) = UF.wavread(inputFile)
    w = get_window(window, M)
    xmX, xpX = stft.stftAnal(x, w, N, H)
    numFrames = int(xmX[:,0].size) #Get number of frames (time slices)
    binFreq = np.arange(N/2+1)*float(fs)/N #Creating array of bin frequencies (positive side only)

    highBandIdx3000 = np.where(binFreq > 1000)[0][0]
    highBandIdx10000 = np.where(binFreq < 1002)[0][-1]

    # calculate energy per band
    engEnv = np.zeros([numFrames])
    for idx_frame in range(numFrames):
        engEnv[idx_frame] = dB2energydB(xmX[idx_frame, highBandIdx3000:highBandIdx10000+1])

    # plt.figure(1, figsize=(9.5, 6))
    #
    # plt.subplot(211)
    # numFrames = int(xmX[:,0].size)
    # frmTime = H*np.arange(numFrames)/float(fs)
    # binFreq = np.arange(N/2+1)*float(fs)/N
    # plt.pcolormesh(frmTime, binFreq, np.transpose(xmX))
    # plt.title('mX (piano.wav), M=1001, N=1024, H=256')
    # plt.autoscale(tight=True)
    #
    # plt.subplot(212)
    # numFrames = int(xmX[:,0].size)
    # frmTime = H*np.arange(numFrames)/float(fs)
    # binFreq = np.arange(N/2+1)*float(fs)/N
    # #plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(xpX),axis=0))
    # #plt.plot(odf[:,0])
    # plt.plot(abs(odf[:,1]))
    # plt.title('ODF adsfsf')
    # plt.autoscale(tight=True)
    #
    # plt.tight_layout()
    # plt.savefig('spectrogram.png')
    # plt.show()

    maxIndex = np.argmax(engEnv)
    timePercent = maxIndex * 1.0 / engEnv.size
    audioLength = x.size / fs
    end_of_chirp = audioLength * timePercent
    return end_of_chirp

# The chirp lasts exactly 14 seconds
# we need to make sure we capture all of it
# If procedure missed some parts of the beginning of the chirp
# that is ok, but we need to account for that
end_of_chirp = findChirpEnd(input_path, 'blackman', 961, 2048, 128)
start_of_chirp = max(0, end_of_chirp - 14.0)

# Round to milliseconds
start_ms = format(start_of_chirp, '.2f')
end_ms = format(end_of_chirp, '.2f')

chirp = AudioSegment.from_wav(input_path)
isolated_chirp = chirp[start_ms:end_ms]
faded_chirp = isolated_chirp.fade_in(2000).fade_out(3000)
