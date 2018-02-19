import tensorflow as tf
import argparse
import os
import sys
import functools
import numpy as np
import math

import calc_features as fe

bad_patient_ids = ["HA001", "HA002"]

# Tensorflow

sample_rate = 4000
frame_length = 512
frame_step = 512
fft_length = 512

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

def get_frequency_spectrum(input, type, region, trial):
    with tf.Session() as sess:
        sess.run(init)
        [db_powers] = sess.run([magnitude], feed_dict={
            audio_file: "%s/%s/%s_%s_%d.wav" % (input, type, type, region, trial),
        })

        freq_spectrum = []
        for i in range(db_powers.shape[0]):
            # https://www.quora.com/How-do-I-convert-Complex-number-output-generated-by-FFT-algorithm-into-Frequency-vs-Amplitude
            frequency = i * math.ceil(sample_rate / fft_length)
            freq_spectrum.append([frequency, db_powers[i]])
        return freq_spectrum

audio_types = ['PS', 'BS', 'TF']
regions = ["LLL", "LML", "LUL", "RLL", "RML", "RUL"]

# TODO: only keep features of least noisey recording
def merge_trials_across_type_and_region(input, type, region, recordings, features):
    return np.mean(features, axis=0)

def get_features(type, spectrum):
    print type
    print spectrum
    if type == 'PS':
        return fe.get_features_for_ps_spectrum(spectrum)
    elif type == 'BS':
        return fe.get_features_for_bs_spectrum(spectrum)
    elif type == 'TF':
        fe.get_features_for_tf_spectrum(spectrum)
    else:
        return []

def process_features_by_type_and_region(input, type, region):
    path = "%s/%s" % (input, type)
    recordings = [file for file in os.listdir(path) if file.startswith("%s_%s" % (type, region))]
    spectrums = [get_frequency_spectrum(input, type, region, trial) for trial in range(1, len(recordings) + 1)]
    features = [get_features(type, spectrum) for spectrum in spectrums]
    return merge_trials_across_type_and_region(input, type, region, recordings, features)

def process_features_by_type(input, type):
    region_features = [process_features_by_type_and_region(input, type, r) for r in regions]
    return np.mean(region_features, axis=0)

def get_features_headers():
    return ['id'] + \
        fe.get_feature_headers_for_ps_spectrum() + \
        fe.get_feature_headers_for_bs_spectrum() + \
        fe.get_feature_headers_for_tf_spectrum()

def process_patient_features(input, patient_id):
    type_features = [process_features_by_type(input, t) for t in audio_types]
    print type_features
    return np.concatenate([patient_id], type_features)
