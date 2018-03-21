import sys
import os
sys.path.append('./analysis')
import general_feature_analysis3 as ga

print "COMPARING BETWEEN CLINICAL AND NOT"
for i in [1,3,5,7]:
	ga.run_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                n_comp_pca = 5,
	                k_neighbors = i,
					include_bs = False,
					include_ps = True,
					include_clinical = False,
					visual_features = ['PS_mean_mfcc_5', 'PS_mean_mfcc_8'], #ga.freq_features,
	                visual_file_1 = 'viz1.png',
	                g1_parameters = ga.g1_params_paper,
	                g2_parameters = ga.g1_params_default)

	ga.run_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                n_comp_pca = 5,
	                k_neighbors = i,
					include_bs = False,
					include_ps = True,
					include_clinical = True,
					visual_features = ['PS_mean_mfcc_5', 'PS_mean_mfcc_8'], #ga.freq_features,
	                visual_file_1 = 'viz1.png',
	                g1_parameters = ga.g1_params_paper,
	                g2_parameters = ga.g1_params_default)

print "COMPARING BETWEEN PS, BS, and PS + BS"
for i in [1,3,5,7]:
	ga.run_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                n_comp_pca = 5,
	                k_neighbors = i,
					include_bs = True,
					include_ps = False,
					include_clinical = True,

	                visual_features = ['PS_mean_mfcc_5', 'PS_mean_mfcc_8'], #ga.freq_features,
	                visual_file_1 = 'viz1.png',
	                g1_parameters = ga.g1_params_paper,
	                g2_parameters = ga.g1_params_default)
	ga.run_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                n_comp_pca = 5,
	                k_neighbors = i,
					include_bs = False,
					include_ps = True,
					include_clinical = True,
					visual_features = ['PS_mean_mfcc_5', 'PS_mean_mfcc_8'], #ga.freq_features,
	                visual_file_1 = 'viz1.png',
	                g1_parameters = ga.g1_params_paper,
	                g2_parameters = ga.g1_params_default)

	ga.run_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                n_comp_pca = 5,
	                k_neighbors = i,
					include_bs = True,
					include_ps = True,
					include_clinical = True,
					visual_features = ['PS_mean_mfcc_5', 'PS_mean_mfcc_8'], #ga.freq_features,
	                visual_file_1 = 'viz1.png',
	                g1_parameters = ga.g1_params_paper,
	                g2_parameters = ga.g1_params_default)
