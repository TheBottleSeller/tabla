./validate.sh

python audio_to_features.py --study ED --input ~/Google\ Drive/Tabla/ED\ Study\ Data/ --output ed_features.csv
python audio_to_features.py --study HA --input ~/Google\ Drive/Tabla/Healthy\ Study\ Data/ --output ha_features.csv
python audio_to_features.py --study PNA --input ~/Google\ Drive/Tabla/Pneumonia\ Study\ Data/ --output ha_features.csv
python concat_csvs.py --in1 ed_features.csv --in2 ha_features.csv --output audio_features.csv
