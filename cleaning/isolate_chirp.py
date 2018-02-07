import os
import sys
import numpy as np
from scipy.signal import get_window
import math
import matplotlib.pyplot as plt
import argparse
from pydub import AudioSegment
from math import floor, ceil
sys.path.insert(0, '../tools')
import stft
import utilFunctions as UF
import sineModel as SM

fade_ms = 500

window = 'blackman' # analysis window type (choice of rectangular, triangular, hanning, hamming, blackman, blackmanharris)
M = 961 # (integer): analysis window size (odd integer value)
N = 2048 # (integer): fft size (power of two, bigger or equal than than M)
H = 128 # (integer): hop size for the STFT computation

parser = argparse.ArgumentParser(description='Clean audio file')
parser.add_argument('--input', type=str, help='input file')
parser.add_argument('--output', type=str, help='output file')
args = parser.parse_args()
input_path = args.input
output_path = args.output

def find_chirp_end_ms_odf():
    def dB2energydB(mdB):
        m = 10 ** (mdB / 20.)
        energy_ = m ** 2.
        #m = 10 * np.log10(m.sum())
        energy_ = 10 * np.log10(np.sum(energy_))
        return energy_

    (fs, x) = UF.wavread(input_path)
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
    return end_of_chirp * 1000

def find_chirp_end_ms_sinusoidal_model():
    window = 'blackmanharris'
    M = 401
    N = 2048
    t = -90
    minSineDur = 0.5
    maxnSines = 30
    freqDevOffset = 3
    freqDevSlope = 0.000000
    H = 50
    minFreq = 100
    maxFreq = 950

    (fs, x) = UF.wavread(input_path)
    w = get_window(window, M)
    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope, minFreq, maxFreq)

    numFrames = int(tfreq[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)

    tfreq[tfreq<=0] = 0
    # tfreq[tfreq<=0] = np.nan
    # plt.plot(frmTime, tfreq)
    # plt.show()

    max = 0
    max_i = 0
    for i in range(0, tfreq.shape[0]):
        sine_max_i = np.argmax(tfreq[i])
        sine_max = tfreq[i][sine_max_i]
        if sine_max > max:
            max = sine_max
            max_i = i

    return frmTime[max_i] * 1000

def process_audio(in, out):
    # The chirp lasts exactly 14 seconds
    # we need to make sure we capture all of it
    # If procedure missed some parts of the beginning of the chirp
    # that is ok, but we need to account for that

    # end_of_chirp_ms = ceil(find_chirp_end_ms_odf())
    end_of_chirp_ms = ceil(find_chirp_end_ms_sinusoidal_model())
    start_of_chirp_ms = floor(max(0, end_of_chirp_ms - 14000))

    chirp = AudioSegment.from_wav(input_path)
    print "Trimming audio: %.2fs - %.2fs" % (start_of_chirp_ms/1000, end_of_chirp_ms/1000)
    isolated_chirp = chirp[start_of_chirp_ms:end_of_chirp_ms]
    print "Fading audio: %dms" % fade_ms
    faded_chirp = isolated_chirp.fade_in(fade_ms).fade_out(fade_ms)

    faded_chirp.export(output_path, format='wav')

process_audio(input_path, output_path)
