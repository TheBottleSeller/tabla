import tensorflow as tf
import argparse
import os
import sys
import functools
import numpy as np
import math
# from pylab import *
# import matplotlib
# import matplotlib.pyplot as plt

import calc_features as fe

sample_rate = 4000
frame_length = 512
frame_step = 512
fft_length = 512

def get_features(patient_dir):
    audio_file = tf.placeholder(tf.string)

    audio_binary = tf.read_file(audio_file)
    waveform = tf.contrib.ffmpeg.decode_audio(
        audio_binary,
        file_format="wav",
        samples_per_second=sample_rate,    # Get Info on .wav files (sample rate)
        channel_count=1             # Get Info on .wav files (audio channels)
    )

    stft = tf.contrib.signal.stft(
        tf.transpose(waveform),
        frame_length,     # frame_lenght, hmmm
        frame_step,     # frame_step, more hmms
        fft_length=fft_length,
        window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=False), # matches audacity
        pad_end=False,
        name="STFT"
    )

    average_fft = tf.reduce_mean(tf.squeeze(stft), 0)

    # https://medium.com/towards-data-science/audio-processing-in-tensorflow-208f1a4103aa
    # magnitude spectrum of positive frequencies in dB
    abs_fft = tf.abs(average_fft)

    def log10(x):
        num = tf.log(x)
        den = tf.log(tf.constant(10, dtype=num.dtype))
        return(tf.div(num, den))

    # Add 1 to ensure all powers are greater than 0 (since taking log base 10)
    magnitude = 20 * log10(abs_fft + 1)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    regions = ["LLL", "LML", "LUL", "RLL", "RML", "RUL"]

    def get_frequency_spectrum(type, region, trial):
        with tf.Session() as sess:
            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run(init)

            [db_powers] = sess.run([magnitude], feed_dict={
                audio_file: "%s/%s/%s_%s_%d.wav" % (patient_dir, type, type, region, trial),
            })

            freq_spectrum = []
            for i in range(db_powers.shape[0]):
                # https://www.quora.com/How-do-I-convert-Complex-number-output-generated-by-FFT-algorithm-into-Frequency-vs-Amplitude
                frequency = i * math.ceil(sample_rate / fft_length)
                freq_spectrum.append([frequency, db_powers[i]])
            return freq_spectrum

    def get_average_frequency_spectrum(type, region):
        recordings_dir = "%s/%s" % (patient_dir, type)
        recordings = [file for file in os.listdir(recordings_dir) if file.startswith("%s_%s" % (type, region))]

        freq_spectrums = []
        for trial in range(1, len(recordings) + 1):
            freq_spectrums.append(get_frequency_spectrum(type, region, trial))

        return np.mean(freq_spectrums, axis=0)

    # Get features for single patient
    # PS features
    ps_region_spectrums = [get_average_frequency_spectrum('PS', region) for region in regions]
    ps_region_features = [fe.get_features_for_ps_spectrum(spectrum) for spectrum in ps_region_spectrums]
    ps_features = np.mean(ps_region_features, axis=0)

    # BS features
    bs_region_spectrums = [get_average_frequency_spectrum('BS', region) for region in regions]
    bs_region_features = [fe.get_features_for_bs_spectrum(spectrum) for spectrum in bs_region_spectrums]
    bs_features = np.mean(bs_region_features, axis=0)

    # TF features
    tf_region_spectrums = [get_average_frequency_spectrum('TF', region) for region in regions]
    tf_region_features = [fe.get_features_for_tf_spectrum(spectrum) for spectrum in tf_region_spectrums]
    tf_features = np.mean(tf_region_features, axis=0)

    headers = fe.get_feature_headers_for_ps_spectrum() + \
        fe.get_feature_headers_for_bs_spectrum() + \
        fe.get_feature_headers_for_tf_spectrum()
    return headers, np.concatenate([ps_features, bs_features, tf_features])
