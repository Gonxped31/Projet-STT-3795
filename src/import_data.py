import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset
import pickle
import os

print("Current Working Directory: ", os.getcwd())

data = load_dataset("common_language", download_mode="reuse_dataset_if_exists")
#print(data['train']) 

#Save 
#with open('dataframes.pkl', 'wb') as handle:
#    pickle.dump(dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Read
#with open('path/to/save/dataframes.pkl', 'rb') as handle:
#    dataframes = pickle.load(handle)

#dataframes['train']['audio'][0]['path']
#for key, df in dataframes.items():
#    df.to_parquet(f'src/data/{key}_dataset.parquet')

with open('dataframes.pkl', 'rb') as handle:
    dataframes = pickle.load(handle)

dataframes['train']
dataframes['test']
dataframes['test']['audio'][0]
print("Current Working Directory: ", os.getcwd())

def save_audio_file(entry, save_directory):
    """
    Saves an audio file from the given entry to the specified directory.

    :param entry: A dictionary with 'bytes' containing the audio data and 'path' the filename.
    :param save_directory: The directory where the audio file will be saved.
    """
    # Extract the audio bytes and the filename from the entry
    audio_bytes = entry['bytes']
    filename = entry['path']
    
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Construct the full path for saving the audio file
    file_path = os.path.join(save_directory, filename)
    
    # Write the audio bytes to the file
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_bytes)

for d in ['train', 'validation', 'test']:
    save_directory = 'wav_files/' + d 
    for entry in dataframes[d]['audio']:
        save_audio_file(entry, save_directory)
        break

dataframes['train'][-1000:-900]

languages = "Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech, Dhivehi, Dutch, English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin, Indonesian, Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mongolian, Persian, Polish, Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish, Tamil, Tatar, Turkish, Ukranian, Welsh".split(", ")
mapper = {i: languages[i] for i in range(len(languages))}
for d in ['train', 'test', 'validation']: 
    dataframes[d]['language'] = dataframes[d]['language'].map(mapper)

import matplotlib.pyplot as plt
split_types = ['train', 'test', 'validation']
for d in split_types:
    print("Central tendency metrics for " + d)
    vector = dataframes[d].groupby('language').count()['path']
    print("Min : " + str(min(vector)))
    print("Max: " + str(max(vector)))
    print("Mean : " + str(np.mean(vector)))
    print("Median : " + str(np.median(vector)))
    plt.figure(figsize=(10, 6))  # Optional: Adjust the size of the plot
    vector.plot(kind='bar')
    plt.title('Bar Chart of column_name')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate x-axis labels to make them readable
    plt.show()
    print("\n")

#Dsitribution are more or uniformally distributed. No need to further randomize the splits

#Idea: Try classification task only with audio files and, then try classification with age and gender and audio file.

#Data processing idea, keep categorical data in csv format