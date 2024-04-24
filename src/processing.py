from common_language import _LANGUAGES

# Data analysis and stats imports
from scipy.fft import fft
from scipy.stats import expon, reciprocal
from scipy.spatial.distance import pdist, squareform
from mutagen.wave import WAVE
from parselmouth.praat import call
import pandas as pd
import numpy as np
import librosa
import librosa.display
import parselmouth
import noisereduce as nr

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# Data visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

# Source for audio dataset:
# https://huggingface.co/datasets/common_language
# https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonLanguage

# WAV file metadata should be consistent on all clips.
# https://mutagen.readthedocs.io/en/latest/api/wave.html#mutagen.wave.WaveStreamInfo
# https://librosa.org/doc/0.10.1/generated/librosa.load.html#librosa.load
channels = 1
bits_per_sample = 16
sample_rate = 16000

# Recommended for speech processing
# https://pypi.org/project/noisereduce/
frame_length = 512

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
    if len(data) < frame_length:
        raise ValueError('Data is too short', len(data), path)
    # clean data
    data = nr.reduce_noise(y=data, sr=sample_rate, n_fft=frame_length)
    return data

def get_dataframe(type):
    df = pd.read_csv(get_path(type, f'{type}_data.csv'))
    #df['Length'] = df['paths'].apply(lambda x: get_length(get_path(type, x)))
    return df

def get_dataframes():
    train_df = get_dataframe('train')
    test_df = get_dataframe('test')
    validation_df = get_dataframe('validation')
    full_df = pd.concat([train_df, test_df, validation_df])
    return full_df, train_df, test_df, validation_df

def get_clean_path(type, name):
    return f'./data/wav_files_clean/{type}/{name}'

def get_clean_dataframe(type):
    df = pd.read_csv(get_clean_path(type, f'{type}_data.csv'))
    #df['Length'] = df['paths'].apply(lambda x: get_length(get_clean_path(type, x)))
    return df

def get_clean_dataframes():
    train_df = get_clean_dataframe('train')
    test_df = get_clean_dataframe('test')
    validation_df = get_clean_dataframe('validation')
    full_df = pd.concat([train_df, test_df, validation_df])
    return full_df, train_df, test_df, validation_df

def get_preprocessed_data():
    train_df = pd.read_csv('./data/train_preprocessed_data.csv')
    test_df = pd.read_csv('./data/test_preprocessed_data.csv')
    validation_df = pd.read_csv('./data/validation_preprocessed_data.csv')
    return pd.concat([train_df, test_df, validation_df])

FEATURE_AUDIO = 'Audio'
FEATURE_MFCCS = 'MFCCs'
FEATURE_SPEC_CENTROID = 'Spec Centroid'
FEATURE_SPEC_BANDWIDTH = 'Spec Bandwidth'
FEATURE_SPEC_CONTRAST = 'Spec Contrast'
FEATURE_SPEC_ROLLOF = 'Spec Rollof'
FEATURE_SPEC_FLATNESS = 'Spec Flatness'
FEATURE_PITCH_TRACK = 'Pitch Track'
FEATURE_FORMANTS = 'Formants'
FEATURE_RMS_ENERGY = 'RMS Energy'
FEATURE_ZCR = 'ZCR'
FEATURE_HNR_MEAN = 'HNR Mean'
    
def get_data_features(path, audio):
    data = get_data(path)

    #Get the attributes
    mfccs = get_Normalized_Mfccs(data)
    specs_measurements = get_spectral_measurements(data)
    pitch_track = get_pitch_sequences(data)
    formants_data = get_formants(path)
    rms_energy = get_rms_energy(data)
    zcr = get_ZCR(data)
    hnr_mean = get_HNR(data)
    
    row = pd.DataFrame({FEATURE_AUDIO: audio ,FEATURE_MFCCS: [np.array(mfccs)], 
                                FEATURE_SPEC_CENTROID: [specs_measurements[0]], FEATURE_SPEC_ROLLOF: [specs_measurements[1]],
                                FEATURE_SPEC_BANDWIDTH: [specs_measurements[2]], FEATURE_SPEC_FLATNESS: [specs_measurements[3]], 
                                FEATURE_SPEC_CONTRAST: [specs_measurements[4]], FEATURE_PITCH_TRACK: [pitch_track],
                                FEATURE_FORMANTS: [formants_data], FEATURE_RMS_ENERGY: [rms_energy],
                                FEATURE_ZCR: [zcr], FEATURE_HNR_MEAN: [hnr_mean]})
    return row

# MFCCs

def get_Normalized_Mfccs(data):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_fft=frame_length, n_mfcc=25)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    mfccs_normalized = ((mfccs.T - mfccs_mean).T) / mfccs_std[:, np.newaxis]
    return mfccs_normalized

# Spectral measurements

def get_spectral_measurements(data):
    spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sample_rate, n_fft=frame_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate, n_fft=frame_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate, n_fft=frame_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=data, n_fft=frame_length)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sample_rate, n_fft=frame_length)
    return (spectral_centroids, spectral_rolloff, spectral_bandwidth, spectral_flatness, spectral_contrast)

# Extract the pitch sequence

def get_pitch_sequences(data):
    pitches, magnitudes = librosa.core.piptrack(y=data, sr=sample_rate, n_fft=frame_length)
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
    # Root Mean Square (RMS) Energy
    return librosa.feature.rms(y=data, frame_length=frame_length)

def get_ZCR(data):
    # Zero-Crossing Rate (ZCR)
    return librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length)

### Voice Quality Features ###
def get_HNR(data):
    # Load the cleaned sound into parselmouth.Sound
    snd = parselmouth.Sound(data, sample_rate)
    # Harmonics-to-Noise Ratio (HNR)
    hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    #print(hnr)
    hnr_mean = call(hnr, "Get mean", 0, 0)
    return hnr_mean

### Modelling ###

### Principal components

def get_PCs(dataframe, percentage_variance):
    print()
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)
    print(f'Scaled_df Mean = {np.mean(scaled_df)},\nScaled_df Std = {np.std(scaled_df)}')


    pca_T = PCA()
    pca_T.fit_transform(scaled_df)
    ev = pca_T.explained_variance_
    print()
    print(f'Total variance = {sum(ev)}')

    pca = PCA(percentage_variance/100)
    principal_components = pca.fit_transform(scaled_df)
    explained_variance = pca.explained_variance_
    percentage = sum(pca.explained_variance_ratio_)
    print(f'Real percentage = {percentage}')
    print(f'Variance for {round(percentage*100, 2)}% = {sum(explained_variance)}')
    print(f'Number of PCs for {round(percentage*100, 2)}% = {len(explained_variance)}')
    print(f'Attribute lost = {len(scaled_df[0]) - len(explained_variance)}')
    names = pca.get_feature_names_out()
    return pd.DataFrame(data=principal_components, columns=names)

# MDS classique
def mds(dataframe, n_components):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)
    print(f'Scaled_df Mean = {np.mean(scaled_df)},\nScaled_df Std = {np.std(scaled_df)}')
    mds = MDS(n_components=n_components, random_state=42, dissimilarity='euclidean')
    mds_transformed = mds.fit_transform(scaled_df)
    return pd.DataFrame(mds_transformed, columns=[f'Component_{i+1}' for i in range(n_components)])

# Mahalanobis distance matrix set up
def compute_mahalanobis_distance_matrix(X):
    # Matrice singuliere a normaliser
    VI = np.linalg.inv(np.cov(X.T) + np.eye(X.shape[1]) * 1e-4)
    mahalanobis_dist = pdist(X, metric='mahalanobis', VI=VI)
    distance_matrix = squareform(mahalanobis_dist)
    return distance_matrix

# MDS with mahalanobis distance
def mds_mahalanobis(dataframe, n_components):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)
    mahalanobis_distance_matrix = compute_mahalanobis_distance_matrix(scaled_df)
    mds = MDS(n_components=n_components, random_state=42, dissimilarity='precomputed')
    mds_transformed = mds.fit_transform(mahalanobis_distance_matrix)
    return pd.DataFrame(mds_transformed, columns=[f'Component_{i+1}' for i in range(n_components)])

