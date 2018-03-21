import numpy as np
import math
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import functools
import csv
from scipy import io
from scipy.io import wavfile
from scipy import stats
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Neil, just make a few alterations to get this program online.
#Sorry for the delay on this project. I tried to get essentia installed,
#and I just figured that I could send you the part of the pipeline, and you
#could link it to the other parts of the data pipeline.

#get an iterable that iterates through all files.
#all_files = #Neil. fill this in.

#get an iterable of all the outcome variable. If you don't
#want this column in the resulting dataframe, just leave it as none.
#all_labels = None

#this is a color dictionary, based on the label of the outcome you want to study
#eg. if you want to look at sound type, create this dictionary:
#color_dictionary = {'BS':'blue','PS':'red','TF':'green'}
#color_dictionary = #Neil, fill this in.

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

abs_fft = tf.abs(average_fft)

def log10(x):
    num = tf.log(x)
    den = tf.log(tf.constant(10, dtype=num.dtype))
    return(tf.div(num, den))

# Add 1 to ensure all powers are greater than 0 (since taking log base 10)
magnitude = 20 * log10(abs_fft + 1)

# Initializing the variables
init = tf.global_variables_initializer()

########### HELPER FUNCTIONS ###############

#this func takes a file and returns a list with the frequency stuff
#Feel free to replace it.
def get_frequency_spectrum(sound_file):
    with tf.Session() as sess:
        sess.run(init)
        [db_powers] = sess.run([magnitude], feed_dict={audio_file: sound_file})
        freq_spectrum = []
        for i in range(db_powers.shape[0]):
            frequency = i * math.ceil(sample_rate / fft_length)
            freq_spectrum.append(db_powers[i])
        return freq_spectrum


#this func just returns the waveform.
def get_waveform(sound_file):
    with tf.Session() as sess:
        sess.run(init)
        [amp] = sess.run([waveform], feed_dict={audio_file: sound_file})
        time_wave = []
        for i in range(amp.shape[0]):
            time_wave.append(amp[i])
        return time_wave


def clean_wave(waveform):
    new = [np.asscalar(x) for x in waveform]
    return(new)


#this just norms the waveform to, basically eliminate the average background
#noise and averages by the volume.
def norm(waveform):
    m = np.mean(waveform)
    sd = np.std(waveform)
    w = [(x-m)/sd for x in waveform]
    return(w)


def train_pca(train_x,
              train_y,
              n_components,
              color_dictionary,
              directory):

    #train pca with n components
    pca = PCA(n_components = n_components)
    pca.fit(train_x)
    reduced = pca.fit_transform(train_x)
    print(pca.explained_variance_ratio_)

    #visualize the training set
    #only show the first two principle components.
    fig, ax = plt.subplots()
    colors = color_dictionary
    result = [colors[x] for x in train_y]
    ax.scatter(reduced[:,0],
               reduced[:,1],
               c = result)
    plt.savefig(directory, bbox_inches='tight')

    #return both the model [0] and the reduced coordinates [1]
    return([pca, reduced])


#######Constructing dataframes with frequencies and waveforms###################

def get_frames_and_models(freq_bins,
                          time_bins,
                          all_files,
                          all_labels,
                          color_dictionary,
                          perc_training,
                          directory
                          ):

    freq_specs = []
    wave_specs = []
    indicies = []

    for index in range(0, len(all_files)):
        try:
            freq_specs.append(get_frequency_spectrum(all_files[index]))
            wave_specs.append(norm(clean_wave(get_waveform(all_files[index]))))
            indicies.append(index)
        #note that the sound files are normalized for volume.
        except:
            next

    all_files = [all_files[x] for x in indicies]
    all_labels = [all_labels[x] for x in indicies]

    #we need to create the dataframes
    #for the frequency and time waveforms
    #it's as clunky af, but it will work.
    speclabel = ['f' + str(x) for x in range(0,freq_bins)]
    freqs = pd.DataFrame(freq_specs, columns=speclabel)

    timelabel = ['t' + str(x) for x in range(0,time_bins)]
    times = pd.DataFrame(wave_specs, columns=timelabel)

    #fconstructing
    files = pd.DataFrame(all_files, columns = ["file_name"])

    #if you don't give me any labels, then you just construct the
    #dataframes.
    if all_labels == None:
        freqs = pd.concat([files, freqs], axis = 1)
        times = pd.concat([files, times], axis = 1)
        data_bundle = {"freqs":freqs, "times":times}
        return data_bundle

    #if you give labels, then set it up for fitting a models.
    if all_labels != None:
        outcomes = pd.DataFrame(all_labels, columns = ["outcome"])
        freqs = pd.concat([files, outcomes, freqs], axis = 1)
        times = pd.concat([files, outcomes, times], axis = 1)
        freqs.to_csv(directory + "freqs.csv")
        times.to_csv(directory + "times.csv")

        nrow = len(freqs.index)
        #permute
        permute = np.random.permutation(range(0,nrow))

        #randomize the rows
        freqs = freqs.iloc[permute,:]
        times = times.iloc[permute,:]
        freqs = freqs.reset_index(drop=True)
        times = times.reset_index(drop=True)

        #get training and testing sets (80/20 split) for frequency spectra
        fnrow = len(freqs.index)
        frowtrain = round(fnrow*perc_training)

        ftrain = freqs.iloc[range(0,frowtrain),:]
        ftrain_x = ftrain.iloc[:,range(2,(freq_bins + 2))]
        ftrain_y = ftrain.outcome

        ftest = freqs.iloc[range(frowtrain,fnrow),:]
        ftest_x = ftest.iloc[:,range(2, (freq_bins + 2))]
        ftest_y = ftest.outcome

        #do the same for the time waves
        tnrow = len(times.index)
        trowtrain = round(tnrow*perc_training)

        ttrain = times.iloc[range(0,trowtrain),:]
        ttrain_x = ttrain.iloc[:,range(2,time_bins + 2)]
        ttrain_y = ttrain.outcome

        ttest = times.iloc[range(trowtrain,tnrow),:]
        ttest_x = ttest.iloc[:,range(2,(time_bins + 2))]
        ttest_y = ttest.outcome

        print("performing PCA on training set")

        print("3 components")
        pca_total = train_pca(ftrain_x, ftrain_y, 3, color_dictionary, directory + "freq_pca_3.png")
        print(pca_total.components_[0])
        print("4 components")
        pca_total1 = train_pca(ftrain_x, ftrain_y, 4, color_dictionary, directory + "freq_pca_4.png")
        print("5 components")
        pca_total2 = train_pca(ftrain_x, ftrain_y, 5, color_dictionary, directory + "freq_pca_5.png")

        pca = pca_total[0]
        reduced_train_x = pca_total[1]
        reduced_test_x = pca.transform(ftest_x)

        pd.DataFrame(freqs).to_csv(directory + "freqs.csv")
        pd.DataFrame(times).to_csv(directory + "times.csv")
        pd.DataFrame(reduced_train_x).to_csv(directory + "freq_pca.csv")
        pd.DataFrame(ftrain_y).to_csv(directory + "outcome.csv")

        data_bundle = {"freqs":freqs,
                       "times":times,
                       "ftrain":ftrain,
                       "ftest":ftest,
                       "ttrain":ttrain,
                       "ttest":ttest,
                       "ftrain_x":ftrain_x,
                       "ftrain_y":ftrain_y,
                       "ftest_x":ftest_x,
                       "ftest_y":ftest_y,
                       "ttrain_x":ttrain_x,
                       "ttrain_y":ttrain_y,
                       "ttest_x":ttest_x,
                       "ttest_y":ttest_y,
                       "pca_3":pca_total,
                       "pca_4":pca_total1,
                       "pca_5":pca_total2,
                       "reduced_train_x":reduced_train_x,
                       "reduced_test_x":reduced_test_x
                       }

        return data_bundle
