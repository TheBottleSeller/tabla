#you might have to change the directory before
#importing this.
import general_feature_analysis3 as ga

#where are the features?
file_in = 'features/features.csv'
#where do you want the results to go?
path_out = 'results/'
#what is the variable being analyzed?
class_id = 'lung_disease'

ga.run_analysis(file_in = file_in, path_out = path_out, class_id = class_id)
