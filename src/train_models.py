# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Data analysis and stats imports
import numpy as np
import pandas as pd
from scipy.stats import expon, reciprocal
from scipy.spatial.distance import pdist, squareform

# Data visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

from common_language import _LANGUAGES
import processing as prlib
import sys

### Modelling ###

def embedded_data():
    full_df, train_df, test_df, validation_df = prlib.get_preprocessed_data()

    X_train, X_test, Y_train, Y_test = train_test_split(full_df.iloc[:, 0:-1], full_df['label'], stratify=full_df['label'], test_size=0.33, random_state=42)
    #X_train, Y_train = df_without_label.iloc[:len(train_df), :], full_df['label'].iloc[:len(train_df)]
    #X_test, Y_test = df_without_label.iloc[len(train_df):, :], full_df['label'].iloc[len(train_df):]

    # PCA
    X_train, embedding = get_PCs(X_train, 95)
    X_test = embedding(X_test)

    # MDS with mahalanobis
    #df_mds = mds_mahalanobis(X_train, 70)
    #df_mds['label'] = df['label']

    return X_train, Y_train, X_test, Y_test

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

    embedding = lambda df: pd.DataFrame(pca.transform(scaler.transform(df)), columns=names)
    return pd.DataFrame(data=principal_components, columns=names), embedding

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

def train_simple(X_train, Y_train, X_test, Y_test):
    # Model initialization
    svm = SVC(verbose=3, random_state=42)
    random_forest = RandomForestClassifier(verbose=3, random_state=42)
    #nn = MLPClassifier(verbose=3)

    # Models fiting
    print("Training SVM")
    svm.fit(X_train, Y_train)
    print("Training RFC")
    random_forest.fit(X_train, Y_train)

    # Models prediction
    svm_predictions = svm.predict(X_test)
    random_forest_predictions = random_forest.predict(X_test)

    svm_accuracy = f1_score(Y_test, svm_predictions, average='macro')
    random_forest_accuracy = f1_score(Y_test, random_forest_predictions, average='macro')

    print(f'SVM accuracy: {svm_accuracy}')
    print(f'RDF accuracy: {random_forest_accuracy}')

param_grid_svm = {
    'C': reciprocal(0.001, 1000),
    'gamma': expon(scale=1.0),
    'kernel': ['linear', 'rbf', 'poly']
}

def train_svm(X_train, Y_train, n_iter):
    svm_clf = SVC()
    random_search_svm = RandomizedSearchCV(svm_clf, param_distributions=param_grid_svm, n_iter=n_iter, verbose=3, cv=5, random_state=42, n_jobs=-1, scoring = 'f1_macro')
    random_search_svm.fit(X_train, Y_train)
    print("Best parameters for SVM:", random_search_svm.best_params_)
    print("Best score:", random_search_svm.best_score_)
    return random_search_svm

param_grid_rf = {
    'n_estimators': [80, 100, 200],
    'max_depth': [3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

def train_rfc(X_train, Y_train, n_iter):
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search_rf = RandomizedSearchCV(rfc, param_distributions=param_grid_rf, n_iter=n_iter, cv=5, verbose=3, random_state=42, n_jobs=-1, scoring = 'f1_macro')
    random_search_rf.fit(X_train, Y_train)
    print("Best parameters:", random_search_rf.best_params_)
    print("Best score:", random_search_rf.best_score_)
    return random_search_rf

def get_metrics(Y_test, predictions):
    accuracy = accuracy_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions, average='macro')
    precision = precision_score(Y_test, predictions, average='macro')
    recall = recall_score(Y_test, predictions, average='macro')
    return ({
        'accuracy_score': accuracy,
        'f1_score': f1,
        'precision_score': precision,
        'recall_score': recall
    })

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = embedded_data()
    n_iter = int(sys.argv[1])
    if n_iter == 0:
        train_simple(X_train, Y_train, X_test, Y_test)
    else:
        train_svm(X_train, Y_train, n_iter)
        #train_rfc(X_train, Y_train, n_iter)