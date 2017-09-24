import argparse
import os
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='Merge two csvs.')
parser.add_argument('--in1', type=str, help='input file 1')
parser.add_argument('--in2', type=str, help='input file 2')
parser.add_argument('--output', type=str, help='output file')
parser.add_argument('--merge_on', type=str, help='key to merge on')
args = parser.parse_args()

in1_path = args.in1
in2_path = args.in2
output = args.output
merge_on = args.merge_on
if not in1_path or not in2_path or not merge_on or not output:
    print('missing parameter')
    sys.exit(-1)

in1 = pd.read_csv(in1_path)
in2 = pd.read_csv(in2_path)
print in1.keys
print in2.keys
print in1.index.tolist()
print in2.index.tolist()
print in1.columns
print in2.columns
merged = in2.merge(in1, on=merge_on, how="outer").fillna("")
merged.to_csv(output, index=False)
