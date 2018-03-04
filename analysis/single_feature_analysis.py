import numpy as np
import scipy as sci
import pandas as pd
import csv
import matplotlib.pyplot as plt

######THE FOLLOWING ARE DEFAULT SETTINGS FOR run_analysis()##########
audio_features = [
                'PS_mean_mfcc_0',
                'PS_mean_mfcc_1',
                'PS_mean_mfcc_2',
                'PS_mean_mfcc_3',
                'PS_mean_mfcc_4',
                'PS_mean_mfcc_5',
                'PS_mean_mfcc_7',
                'PS_mean_mfcc_8',
                'PS_mean_mfcc_9',
                'PS_mean_mfcc_10',
                'PS_mean_mfcc_11',
                'PS_mean_centroid',
                # 'BS_mean_mfcc_0',
                # 'BS_mean_mfcc_1',
                # 'BS_mean_mfcc_2',
                # 'BS_mean_mfcc_3',
                # 'BS_mean_mfcc_4',
                # 'BS_mean_mfcc_5',
                # 'BS_mean_mfcc_7',
                # 'BS_mean_mfcc_8',
                # 'BS_mean_mfcc_9',
                # 'BS_mean_mfcc_10',
                # 'BS_mean_mfcc_11',
                # 'BS_mean_centroid',
                ]

clinical_features = [
    'age',
    'male',
    'female',
    'thorax_circ',
    'smoking_packs',

    # 'temp',
    # 'bp_systolic',
    # 'bp_diastolic',
    # 'hr',
    # 'rr',
    # 'sp02',
    # 'peak_flow',
    # 'sob',
    # 'wheezing'
]

freq_features = audio_features + clinical_features

visual_sample = ['PS_mean_mfcc_5',
                'PS_mean_mfcc_8']

#####################################################################

#This is an Analysis class that
#will structure further statistical analyses.
#enter the data as a dataframe.
#build models and put them in Analysis.models dictionary

class Analysis():
    def __init__(self, data, class_id, models ={}):
        self.data = data
        self.class_id = class_id
        self.models = models

    #this plots a group of features against themselves.
    # def scatter_plot(self, features, directory):
    #     colors = ['r', 'g', 'c', 'b', 'k', 'm', 'y']
    #     plt.figure(figsize= (10,10))
    #     for ii in range(len(features)):
    #         for jj in range(len(features)):
    #             plt.subplot(len(features),len(features), ii*len(features) + jj + 1)
    #             for i in range(self.data.shape[0]):
    #                 x_cord = self.data.loc[i,features[jj]]
    #                 y_cord = self.data.loc[i,features[ii]]
    #                 plt.scatter(x_cord,y_cord, c = colors[self.data.loc[i,self.class_id]], s=20, alpha=0.75)
    #                 #circ = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor=colors[])
    #                 #legArray.append(circ)
    #                 if jj == 0:
    #                     plt.ylabel(features[ii])
    #                 else:
    #                     plt.yticks([])
    #                 if ii == len(features) -1:
    #                     plt.xlabel(features[jj])
    #                 else:
    #                     plt.xticks([])
    def scatter_plot(self, features, directory):
        x_cord = self.data.loc[3,features[3]]
        y_cord = self.data.loc[3,features[3]]
        plt.scatter(x_cord,y_cord, c = colors[self.data.loc[3,self.class_id]], s=20, alpha=0.75)
        plt.ylabel(features[3])
        plt.xlabel(features[3])
        plt.savefig(directory)

def run_single_analysis(file_in = '/Users/samuelzetumer/Desktop/tabla-master/features/features.csv',
                path_out = '/Users/samuelzetumer/Desktop/tabla-master/analysis/',
                class_id = 'lung_disease',
                features = freq_features,
                visual_features = visual_sample,
                visual_file_1 = 'viz1.png'):

    df = pd.read_csv(file_in)
    a1 = Analysis(data = df, class_id = class_id)



    #destinations!
    visualize_path = path_out + visual_file_1


    #now it needs to plot some stuff.
    a1.scatter_plot(visual_features, visualize_path)

def f_ttest(col1, col2):
    result = sci.stats.ttest_ind(col1, col2, equal_var = False)
    return(result[1])
