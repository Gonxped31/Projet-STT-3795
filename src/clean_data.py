from common_language import _LANGUAGES
import processing as prlib

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import soundfile as sf

def clean_sound(audio): 
    tresh = 1000
    first_non_zero_index = np.where(np.abs(audio) > tresh)[0][0]
    last_non_zero_index = np.where(np.abs(audio[::-1]) > tresh)[0][0]
    trimmed_audio_data = audio[first_non_zero_index:-last_non_zero_index]
    return trimmed_audio_data

def clean_file(type, wav_name):
    audio = prlib.get_data(prlib.get_path(type, wav_name))
    clean_audio = prlib.clean_sound(audio)
    audio_data_int16 = clean_audio.astype(np.int16)
    #wav_name = wav_name.split("_")[2] + "_" + str(j) + ".wav"
    #curr_df.loc[j, "new_paths"] = wav_name
    output_filename = prlib.get_clean_path(type, wav_name)
    sf.write(output_filename, audio_data_int16, prlib.sample_rate)

if __name__ == '__main__':
    # Clean silence from beginning and end of audio files
    _, train, test, validation = prlib.get_dataframes()

    for (type, curr_df) in [("train", train), ("test", test), ("validation", validation)]:
        #curr_df['new_paths'] = np.nan
        for j in range(len(curr_df)):
            wav_name = str(curr_df['paths'][j])
            try:
                clean_file(type, wav_name)
            except Exception as e:
                print(e)
