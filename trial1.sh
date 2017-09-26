#!/bin/sh

set -e

./gen_patient_features.sh

python normalize_features.py \
  --input features.csv \
  --cols 0,1,2,3,4,5,6,7,8,9,10,20 \
  --output normalized_features_trial_1.csv

python train_nn.py --input normalized_features_trial_1.csv
