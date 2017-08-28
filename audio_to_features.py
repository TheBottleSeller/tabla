import tensorflow as tf
import argparse
import os
import sys
import functools
import numpy as np

# parser = argparse.ArgumentParser(description='Validate pnuemonia audio files.')
# parser.add_argument('--audio_file_path', type=str, help='file path to audio files')
# parser.add_argument('--study', type=str, help='ED or PNA')
# args = parser.parse_args()
#
# study = args.study
# audio_file_path = args.audio_file_path.rstrip('/')
# if not audio_file_path:
#     print('audio file path is required')
#     sys.exit(-1)
#
# if not study == 'PNA' and not study == 'ED':
#     print('study must be either PNA or ED')
#     sys.exit(-1)
#
# patient_dirs = [file for file in os.listdir(audio_file_path) if file.startswith(study)]
# patient_dirs.sort()

# raw_binary_data=tf.placeholder(tf.)

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

magnitude = 20 * log10(tf.maximum(abs_fft, 1E-06))

# def angle(z):
#     if z.dtype == tf.complex128:
#         dtype = tf.float64
#     elif z.dtype == tf.complex64:
#         dtype = tf.float32
#     else:
#         raise ValueError('input z must be of type complex64 or complex128')
#
#     x = tf.real(z)
#     y = tf.imag(z)
#     x_neg = tf.cast(x < 0.0, dtype)
#     y_neg = tf.cast(y < 0.0, dtype)
#     y_pos = tf.cast(y >= 0.0, dtype)
#     offset = x_neg * (y_pos - y_neg) * np.pi
#     return tf.atan(y / x) + offset
#
# # phase of positive frequencies
# phase = angle(average_fft)

# fft = tf.cast(tf.spectral.fft(
#     tf.cast(waveform, tf.complex64),
#     name="FFT"
# ), tf.float32)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(init)

    [sftfval, output, m] = sess.run([stft, average_fft, magnitude], feed_dict={
        audio_file: "/Users/neilbatlivala/Google Drive/Tabla/Pneumonia Study Data/PNA002/PS/PS_LLL_1.wav"
    })

    # print sftfval
    print m

    # print sftfval.shape
    # print output
    file = open("fft.csv", "w")

    for i in range(m.shape[0]):
        # https://www.quora.com/How-do-I-convert-Complex-number-output-generated-by-FFT-algorithm-into-Frequency-vs-Amplitude
        frequency = i * sample_rate / fft_length
        file.write('%f,%f' % (frequency, m[i]))
        file.write(',\n')
    # print stftval.length
print "done"
