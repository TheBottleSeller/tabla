import os
import sys
import numpy as np
from scipy.signal import get_window
import math
import matplotlib.pyplot as plt

sys.path.insert(0, '../tools')
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

def dB2energydB(mdB):
    m = 10 ** (mdB / 20.)
    energy_ = m ** 2.

    #m = 10 * np.log10(m.sum())
    energy_ = 10 * np.log10(np.sum(energy_))

    return energy_

def findChirpEnd(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming,
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF
            computed on the low frequency band and the second column is the ODF computed on the high
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """

    (fs, x) = UF.wavread(inputFile)

    w = get_window(window,M)

    xmX, xpX = stft.stftAnal(x, w, N, H)

    #Get number of frames (time slices)
    numFrames = int(xmX[:,0].size)

    #Creating array of bin frequencies (positive side only)
    binFreq = np.arange(N/2+1)*float(fs)/N

    #Locate the first bin frequency index and last bin frequency index in both desired ranges

    lowBandIdx0 = np.where(binFreq > 40)[0][0]

    lowBandIdx3000 = np.where(binFreq < 3000)[0][-1]

    highBandIdx3000 = np.where(binFreq > 1000)[0][0]

    highBandIdx10000 = np.where(binFreq < 1002)[0][-1]

    # calculate energy per band
    engEnv = np.zeros([numFrames, 2])
    for idx_frame in range(numFrames):
        engEnv[idx_frame, 0] = dB2energydB(xmX[idx_frame,lowBandIdx0:lowBandIdx3000+1])
        engEnv[idx_frame, 1] = dB2energydB(xmX[idx_frame, highBandIdx3000:highBandIdx10000+1])

    odf = np.zeros(engEnv.shape)
    for i in range(0, engEnv.shape[0]):
        if i == 0:
            odf[0] = [0, 0]
        else:
            odf[i][0] = engEnv[i][0] - engEnv[i-1][0]
            odf[i][1] = engEnv[i][1] - engEnv[i-1][1]

    plt.figure(1, figsize=(9.5, 6))

    plt.subplot(211)
    numFrames = int(xmX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)
    binFreq = np.arange(N/2+1)*float(fs)/N
    plt.pcolormesh(frmTime, binFreq, np.transpose(xmX))
    plt.title('mX (piano.wav), M=1001, N=1024, H=256')
    plt.autoscale(tight=True)

    plt.subplot(212)
    numFrames = int(xmX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)
    binFreq = np.arange(N/2+1)*float(fs)/N
    #plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(xpX),axis=0))
    #plt.plot(odf[:,0])
    plt.plot(abs(odf[:,1]))
    plt.title('ODF adsfsf')
    plt.autoscale(tight=True)

    plt.tight_layout()
    plt.savefig('spectrogram.png')
    plt.show()

    #maxIndex = np.argmax(abs(odf[:,1]))
    maxIndex = np.argmax(engEnv[:,1])
    timePercent = maxIndex * 1.0 / odf[:,1].size
    audioLength = x.size / fs
    print timePercent
    end_of_chirp = audioLength * timePercent
    print audioLength
    return end_of_chirp

end_of_chirp = findChirpEnd('./chirp.wav', 'blackman', 961, 2048, 128)
print end_of_chirp

# The chirp lasts exactly 14 seconds
# we need to make sure we capture all of it
# If procedure missed some parts of the beginning of the chirp
# that is ok, but we need to account for that
start_of_chirp = max(0, end_of_chirp - 14.0)
