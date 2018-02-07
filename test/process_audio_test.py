import argparse
import sys
sys.path.insert(0, '../')
from process_audio import process_audio

parser = argparse.ArgumentParser(description='Clean audio file')
parser.add_argument('--input', type=str, help='input file')
parser.add_argument('--output', type=str, help='output file')
args = parser.parse_args()

process_audio('PS', 'chirp.wav', 'chirp_processed.wav')
