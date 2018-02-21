#you might have to change the directory before
#importing this.
import sys
import os
sys.path.append('./analysis')
import general_feature_analysis3 as ga

ga.run_analysis(file_in = 'features/features.csv',
                path_out = 'analysis/',
                class_id = 'lung_disease',
                features = ga.freq_features,
                n_comp_pca = 4,
                k_neighbors = 5,
                visual_features = ga.visual_sample,
                visual_file_1 = 'viz1.png')
