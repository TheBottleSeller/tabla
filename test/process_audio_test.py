import sys

sys.path.append('../')
sys.path.append('../tools')
from process_audio import process_audio

process_audio(
    'PS',
    '../raw_data/heart_failure/HF001/HF001_011918/PS/PS_LLL_2.wav',
    '../processed_data/heart_failure/HF001/HF001_011918/PS/PS_LLL_2.wav'
)
