from common_language import _LANGUAGES
import processing as prlib

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import soundfile as sf

def clean_sound(audio):
    tresh = 0.005
    first_non_zero_index = np.where(np.abs(audio) > tresh)[0][0]
    last_non_zero_index = np.where(np.abs(audio[::-1]) > tresh)[0][0]
    trimmed_audio_data = audio[first_non_zero_index:-last_non_zero_index]
    return trimmed_audio_data

def clean_file(type, wav_name):
    input_filepath = prlib.get_path(type, wav_name)
    audio = prlib.get_data(input_filepath)
    clean_audio = clean_sound(audio)
    output_filepath = prlib.get_clean_path(type, wav_name)
    sf.write(output_filepath, clean_audio, samplerate=prlib.sample_rate, subtype='PCM_16')

if __name__ == '__main__':
    # Clean silence from beginning and end of audio files
    _, train, test, validation = prlib.get_dataframes()

    for (type, curr_df) in [("train", train), ("test", test), ("validation", validation)]:
        #curr_df['new_paths'] = np.nan
        for j in range(len(curr_df)):
            wav_name = str(curr_df['paths'][j])
            print(j, "/", len(curr_df), wav_name)
            try:
                clean_file(type, wav_name)
            except Exception as e:
                print(e)
