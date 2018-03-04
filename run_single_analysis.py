import sys
import os
sys.path.append('./analysis')
import single_feature_analysis as sf

#for i in range(1, 8):
#	print "KNN: %d" % i
sf.run_single_analysis(file_in = 'features/features.csv',
	                path_out = 'analysis/',
	                class_id = 'lung_disease',
	                features = sf.freq_features,
	                visual_features = sf.visual_sample,
	                visual_file_1 = 'viz1.png')