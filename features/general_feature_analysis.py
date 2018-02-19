import numpy as np
import scipy as sci
import pandas as pd
import csv

#an "Analysis" requires the data to be analyzed,
#and the name of the column that you want to be the
#result or class that you would like to analyze the
#rest of the data with respect to.

#it can take as many statistical tests you would like
#to apply, as well as a dataframe of any size.
class Analysis():
    def __init__(self, data, class_id):
        self.data = data
        self.class_id = class_id
        
    def single_variate_test(self, func):
        df0 = df[df[self.class_id] == 0]
        df1 = df[df[self.class_id] == 1]
        result_df = pd.DataFrame()
        for col in list(self.data):
            try:
                result = func(df0[col],df1[col])
                print(result)
                result_df[col] = [result]
            except:
                result_df[col] = [None]
                print("In none")
        return(result_df)
        

#All functions fed into single_variate_test must take two arrays
#and return the result of the analysis (ie, the p-value, or
#whatever)

#Eg... Ttests:
#formatted t test (f_ttest)
#takes the two arrays
#returns a p-value for the ttest
def f_ttest(col1, col2):
    result = sci.stats.ttest_ind(col1, col2, equal_var = False)
    return(result[1])

####Example

#this is the sample file
df = pd.read_csv('/Users/samuelzetumer/Desktop/tabla-master/features/features.csv')

test = Analysis(data = df, class_id = 'lung_disease')
result = test.single_variate_test(f_ttest)
