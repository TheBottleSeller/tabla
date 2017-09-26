import argparse
import os
import sys
import pandas as pd
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Merge two csvs.')
parser.add_argument('--input', type=str, help='input file')
parser.add_argument('--cols', type=str, help='columns')
parser.add_argument('--output', type=str, help='output file')
args = parser.parse_args()

input = args.input
output = args.output
cols = args.cols.split(',')
cols = [int(id) for id in cols]

if not input or not output or len(cols) == 0:
    print('missing parameters')
    sys.exit(-1)

input = pd.read_csv(input, usecols=cols)

print "STARTED NORMALIZING FEATURES"
print list(input.columns.values)

min_max_scaler = preprocessing.MinMaxScaler()
input_scaled = min_max_scaler.fit_transform(input)
normalized_input = pd.DataFrame(input_scaled)
normalized_input.to_csv(output, index=False, header=False)

print "FINISHED NORMALIZING FEATURES"
