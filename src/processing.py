from common_language import _LANGUAGES
from scipy.fft import fft
from mutagen.wave import WAVE
from parselmouth.praat import call
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
import parselmouth
import noisereduce as nr

# Source for audio dataset:
# https://huggingface.co/datasets/common_language
# https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonLanguage

# WAV file metadata should be consistent on all clips.
# https://mutagen.readthedocs.io/en/latest/api/wave.html#mutagen.wave.WaveStreamInfo
# https://librosa.org/doc/0.10.1/generated/librosa.load.html#librosa.load
channels = 1
bits_per_sample = 16
sample_rate = 16000

frame_length = 2048
hop_length = 512

def get_path(type, name):
    return f'./data/wav_files/{type}/{name}'

def get_length(path):
    audio = WAVE(path)
    audio_info = audio.info
    assert(audio_info.bits_per_sample == bits_per_sample)
    assert(audio_info.sample_rate == sample_rate)
    assert(audio_info.channels == channels)
    return audio_info.length

def get_data(path):
    data, sr = librosa.load(path, sr=None)
    assert(sr == sample_rate)
    return data

def get_dataframe(type):
    df = pd.read_csv(get_path(type, f'{type}_data.csv'))
    df['Length'] = df['paths'].apply(lambda x: get_length(get_path(type, x)))
    return df

def get_data_vector(type, audio):
    path = get_path(type, audio)
    data = get_data(path)
    #clean data: https://pypi.org/project/noisereduce/
    data = nr.reduce_noise(y=data, sr=sample_rate)

    #Get the attributes
    mfccs = get_Normalized_Mfccs(data)
    specs_measurements = get_spectral_measurements(data)
    pitch_track = get_pitch_sequences(data)
    formants_data = get_formants(path)
    rms_energy = get_rms_energy(data)
    zcr = get_ZCR(data)
    hnr_mean = get_HNR(data)
    
    row = pd.DataFrame({'Audio': audio ,'MFCCs': [np.array(mfccs)], 
                                'Spec Centroid': [specs_measurements[0]], 'Spec Rollof': [specs_measurements[1]],
                                'Spec Bandwidth': [specs_measurements[2]], 'Spec Flatness': [specs_measurements[3]], 
                                'Spec Contrast': [specs_measurements[4]], 'Pitch Track': [pitch_track],
                                'Formants': [formants_data], 'RMS Energy': [rms_energy],
                                'ZCR': [zcr], 'HNR Mean': [hnr_mean]})
    return row

# MFCCs

def get_Normalized_Mfccs(data):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=25)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    mfccs_normalized = ((mfccs.T - mfccs_mean).T) / mfccs_std[:, np.newaxis]
    return mfccs_normalized

# Spectral measurements

def get_spectral_measurements(data):
    spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=data)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sample_rate)
    return (spectral_centroids, spectral_rolloff, spectral_bandwidth, spectral_flatness, spectral_contrast)

# Extract the pitch sequence

def get_pitch_sequences(data):
    pitches, magnitudes = librosa.core.piptrack(y=data, sr=sample_rate)
    # Select the dominant pitch at each frame
    pitch_track = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_track.append(pitch)

    pitch_track = np.array(pitch_track)

    # Remove zeros values (unvoiced frames)
    pitch_track = pitch_track[pitch_track > 0]
    return pitch_track

### Get formants data ###
def get_formants(path):
    audio = parselmouth.Sound(path)
    formants = audio.to_formant_burg()
    number_points = int(audio.duration / 0.01) + 1
    formant_data = {'time': [], 'F1': [], 'F2': [], 'F3': []}
    for i in range(number_points):
        time = i * 0.01
        formant_data['time'].append(time)
        formant_data['F1'].append(formants.get_value_at_time(1, time))
        formant_data['F2'].append(formants.get_value_at_time(2, time))
        formant_data['F3'].append(formants.get_value_at_time(3, time))

    return formant_data

### Energy and Amplitude Features ###

def get_rms_energy(data):
    # Root Mean Square (RMS) Energy - with a frame length of 2048 (default)
    return librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)

def get_ZCR(data):
    # Zero-Crossing Rate (ZCR) - with a frame length of 2048 (default)
    return librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)

### Voice Quality Features ###
def get_HNR(data):
    # Load the cleaned sound into parselmouth.Sound
    snd = parselmouth.Sound(data, sample_rate)
    # Harmonics-to-Noise Ratio (HNR)
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    #print(hnr)
    hnr_mean = call(hnr, "Get mean", 0, 0)
    return hnr_mean