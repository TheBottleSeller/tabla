import argparse
import os
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='Merge two csvs.')
parser.add_argument('--in1', type=str, help='input file 1')
parser.add_argument('--in2', type=str, help='input file 2')
parser.add_argument('--output', type=str, help='output file')
args = parser.parse_args()

in1_path = args.in1
in2_path = args.in2
output = args.output
if not in1_path or not in2_path or not output:
    print('missing parameter')
    sys.exit(-1)

in1 = pd.read_csv(in1_path)
in2 = pd.read_csv(in2_path)
out = pd.concat([in1, in2], ignore_index=False)
print out
out.to_csv(output, index=False)
