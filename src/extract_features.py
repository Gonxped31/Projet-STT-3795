from common_language import _LANGUAGES
import processing as prlib

from scipy.fft import fft
from mutagen.wave import WAVE
from parselmouth.praat import call
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
import parselmouth
import noisereduce as nr
import os 

def extract_zcr_features(zcr_vector, hnr_mean): 
    # Calculate aggregated features
    features = {
        "mean_zcr": np.mean(zcr_vector),
        "std_dev_zcr": np.std(zcr_vector),
        "median_zcr": np.median(zcr_vector),
        "min_zcr": np.min(zcr_vector),
        "max_zcr": np.max(zcr_vector),
        "q25_zcr": np.percentile(zcr_vector, 25),
        "q75_zcr": np.percentile(zcr_vector, 75),
        "iqr_zcr": np.percentile(zcr_vector, 75) - np.percentile(zcr_vector, 25),
        "hnr_mean": hnr_mean
    }

    # Convert the features dictionary to a pandas DataFrame
    features_df = pd.DataFrame([features])
    
    return features_df    

def extract_rms_features(rms_energy): 
    features = {
            "mean_energy": np.mean(rms_energy),
            "std_dev_energy": np.std(rms_energy),
            "median_energy": np.median(rms_energy),
            "min_energy": np.min(rms_energy),
            "max_energy": np.max(rms_energy),
            "energy_range": np.max(rms_energy) - np.min(rms_energy),
            "q25_energy": np.percentile(rms_energy, 25),
            "q75_energy": np.percentile(rms_energy, 75),
            "energy_iqr": np.percentile(rms_energy, 75) - np.percentile(rms_energy, 25),
            "energy_variability": np.sum(np.abs(np.diff(rms_energy))),
            "zero_crossing_rate": np.sum(np.diff(np.sign(np.diff(rms_energy))) != 0) / len(rms_energy),
            "low_energy_frame_rate": np.sum(rms_energy < (0.5 * np.mean(rms_energy))) / len(rms_energy)
        }

    features_df = pd.DataFrame([features])

    return features_df


def extract_formants(f1, f2, f3): 
    f1, f2, f3 = map(lambda x: np.nan_to_num(np.asarray(x)), [f1, f2, f3])
    features = {}
    for formant, name in zip([f1, f2, f3], ['F1', 'F2', 'F3']):
        features[f'{name}_mean'] = np.nanmean(formant)
        features[f'{name}_std_dev'] = np.nanstd(formant)
        features[f'{name}_median'] = np.nanmedian(formant)
        features[f'{name}_min'] = np.nanmin(formant)
        features[f'{name}_max'] = np.nanmax(formant)
        features[f'{name}_range'] = np.nanmax(formant) - np.nanmin(formant)
        features[f'{name}_q25'] = np.nanpercentile(formant, 25)
        features[f'{name}_q75'] = np.nanpercentile(formant, 75)
        features[f'{name}_iqr'] = np.nanpercentile(formant, 75) - np.nanpercentile(formant, 25)
        # Ensure there are at least 2 elements to compute diff, otherwise default to 0
        features[f'{name}_delta_sum'] = np.sum(np.abs(np.diff(formant))) if len(formant) > 1 else 0

    if all(len(formant) > 0 for formant in [f1, f2, f3]):
        features['F2_F1_diff_mean'] = np.nanmean(f2 - f1)
        features['F3_F2_diff_mean'] = np.nanmean(f3 - f2)
    else:
        features['F2_F1_diff_mean'], features['F3_F2_diff_mean'] = 0, 0

    features_df = pd.DataFrame([features])
    return features_df

def extract_spectre(data, type):
    # Define feature names
    feature_names = ['mean_' + type, 'std_' + type, 'median_' + type, 'min_' + type, 'max_' + type, 'q25_' + type, 'q75_' + type]
    
    # Compute the features
    mean_val = np.mean(data)
    std_val = np.std(data)
    median_val = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    q25, q75 = np.percentile(data, [25, 75])
    
    # Collect features into a list
    features = [mean_val, std_val, median_val, min_val, max_val, q25, q75]
    
    # Create a DataFrame from the features list
    features_df = pd.DataFrame([features], columns=feature_names)
    
    return features_df

# Outputs 3 features per band (Contrast peak, temporal evolution, rate of change) = 21 
# Mean and std of spectral contrast
def extract_contrast(spectral_contrast):
    features = []
    feature_names = []

    # Iterate over each frequency band to calculate band-specific features
    for band in range(spectral_contrast.shape[0]):
        contrast_band = spectral_contrast[band, :]

        # Count significant peaks
        peaks, _ = find_peaks(contrast_band, height=np.mean(contrast_band))
        features.append(len(peaks))
        feature_names.append(f'band_{band}_peaks')

        # Temporal evolution: difference between means of the first and second halves
        mid_point = len(contrast_band) // 2
        mean_diff = np.mean(contrast_band[mid_point:]) - np.mean(contrast_band[:mid_point])
        features.append(mean_diff)
        feature_names.append(f'band_{band}_mean_diff')

        # Rate of change (derivative)
        derivative = np.mean(np.abs(np.diff(contrast_band)))
        features.append(derivative)
        feature_names.append(f'band_{band}_derivative')

    # Add overall statistical measures for the entire spectral contrast matrix
    overall_mean = np.mean(spectral_contrast)
    features.append(overall_mean)
    feature_names.append('overall_mean')

    overall_std = np.std(spectral_contrast)
    features.append(overall_std)
    feature_names.append('overall_std')

    # Convert the features list into a DataFrame
    features_df = pd.DataFrame([features], columns=feature_names)

    return features_df

def mfccs_to_df(mfcc_means,mfcc_stds): 
    # Ensure mfcc_means and mfcc_stds are flat arrays
    mfcc_means = mfcc_means.flatten()
    mfcc_stds = mfcc_stds.flatten()
    
    # Generate column names
    mean_col_names = [f'MFCC_mean_{i+1}' for i in range(len(mfcc_means))]
    std_col_names = [f'MFCC_std_{i+1}' for i in range(len(mfcc_stds))]
    
    # Combine the MFCC means and stds into a single DataFrame
    mfcc_features_df = pd.DataFrame([np.concatenate([mfcc_means, mfcc_stds])],
                                    columns=mean_col_names + std_col_names)
    
    return mfcc_features_df

def extract_feature_row(path, label):
    data = prlib.get_data(path)

    #Get the attributes
    mfccs = prlib.get_Normalized_Mfccs(data)
    specs_measurements = prlib.get_spectral_measurements(data)
    pitch_track = prlib.get_pitch_sequences(data)
    formants_data = prlib.get_formants(path)
    rms_energy = prlib.get_rms_energy(data)
    zcr = prlib.get_ZCR(data)
    hnr_mean = prlib.get_HNR(data)

    # mfccs: Get mean and std atributes
    mfccs = mfccs_to_df(np.mean(mfccs, axis = 1),np.std(mfccs, axis = 1))

    # Spec measurements: Use extract specter to produce features
    spectre_centroid_df = extract_spectre(specs_measurements[0], "centroid")
    spectre_rollof_df = extract_spectre(specs_measurements[1], "rollof")
    spectre_bandwidth_df = extract_spectre(specs_measurements[2], "bandwidth")
    spectre_flatness_df = extract_spectre(specs_measurements[3], "flatness")
    

    # Spectre contrast
    spectre_contrast_df = extract_contrast(specs_measurements[4])

    # Pitch track: use extract spectre method
    # Add IQR and pitch delta sum -> 9 features
    #print(pitch_track.shape)
    pitch_track_df = extract_spectre(pitch_track, "pitch_track")
    # Formants : Call extract_formants with f1, f2, f3
    # Returns dataframe with 32 features
    formants_df = extract_formants(formants_data["F1"], formants_data["F2"], formants_data["F3"])
    
    # RMS
    rms_energy_df = extract_rms_features(rms_energy[0])

    #ZCR and HNR mean
    zcr_hnr = extract_zcr_features(zcr, hnr_mean)

    combined_features_row = pd.concat([mfccs, spectre_centroid_df,\
                                    spectre_rollof_df, \
                                        spectre_bandwidth_df,\
                                            spectre_flatness_df, \
                                                spectre_contrast_df,\
                                                    pitch_track_df,\
                                                        formants_df, \
                                                            rms_energy_df, zcr_hnr, label], axis = 1)

    """
    row = pd.DataFrame({'Audio': audio ,'MFCCs': [np.array(mfccs)],
                                'Spec Centroid': [specs_measurements[0]], 'Spec Rollof': [specs_measurements[1]],
                                'Spec Bandwidth': [specs_measurements[2]], 'Spec Flatness': [specs_measurements[3]],
                                'Spec Contrast': [specs_measurements[4]], 'Pitch Track': [pitch_track],
                                'Formants': [formants_data], 'RMS Energy': [rms_energy],
                                'ZCR': [zcr], 'HNR Mean': [hnr_mean]})
    """
    return combined_features_row

def extract_features(type, df):
    audios = df['paths'].tolist()
    attributes_df = pd.DataFrame()
    for j in range(len(audios)):
        print(j, "/", len(audios), ":", audios[j])
        path = prlib.get_clean_path(type, audios[j])
        print(audios[j] + " " + path)
        try:
            label = pd.DataFrame({ 'label': [df['language'][j]] })
            combined_features_row = extract_feature_row(path, label)
            attributes_df = pd.concat([attributes_df, combined_features_row], ignore_index=True)
            # attributes_df['label'] = df['language'][j]
        except Exception as e:
            print(e)
    return attributes_df

if __name__ == '__main__':
    # Extract features from the dataset
    _, train_df, test_df, validation_df = prlib.get_clean_dataframes()
    for (type, df) in [('train', train_df), ('test', test_df), ('validation', validation_df)]:
        attributes_df = extract_features(type, df)
        attributes_df.to_csv(f'./data/{type}_preprocessed_data.csv', index=False) # , sep=',', encoding='utf-8', index=False)
