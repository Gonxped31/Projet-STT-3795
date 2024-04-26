# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

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
    #X_train, embedding = mds(X_train, 90, compute_mahalanobis_distance_matrix)
    #X_test = embedding(X_test)

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

    pca = PCA(percentage_variance/100, verbose=3)
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
def mds(dataframe, n_components, dissimilarity=True):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(dataframe)
    print(f'Scaled_df Mean = {np.mean(scaled_df)},\nScaled_df Std = {np.std(scaled_df)}')

    x_train = scaled_df if dissimilarity else compute_mahalanobis_distance_matrix(scaled_df)
    dissimilarity = 'euclidean' if dissimilarity else 'precomputed'

    mds = MDS(n_components=n_components, random_state=42, dissimilarity=dissimilarity, verbose=3)
    mds_transformed = mds.fit_transform(x_train)
    
    names = mds.get_feature_names_out()
    embedding = lambda df: pd.DataFrame(mds.transform(scaler.transform(df)), columns=names)
    return pd.DataFrame(mds_transformed, columns=names), embedding

# Mahalanobis distance matrix set up
def compute_mahalanobis_distance_matrix(X):
    # Matrice singuliere a normaliser
    VI = np.linalg.inv(np.cov(X.T) + np.eye(X.shape[1]) * 1e-4)
    mahalanobis_dist = pdist(X, metric='mahalanobis', VI=VI)
    distance_matrix = squareform(mahalanobis_dist)
    return distance_matrix

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
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
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
    type = sys.argv[2]

    # Model initialization
    if n_iter == 0 and type == "svc":
        model = SVC(random_state=42)
    elif n_iter == 0 and type == "rfc":
        model = RandomForestClassifier(verbose=3, random_state=42, \
                                        criterion='entropy', max_depth=None, max_features=None, min_samples_leaf=2, min_samples_split=2, n_estimators=1000)
        #nn = MLPClassifier(verbose=3)
        print("Training...")
        model.fit(X_train, Y_train)
    elif type == "svc":
        model = train_svm(X_train, Y_train, n_iter)
    elif type == "rfc":
        model = train_rfc(X_train, Y_train, n_iter)
    predictions = model.predict(X_test)
    accuracy = get_metrics(Y_test, predictions)
    print(f'Model accuracy: {accuracy}')
