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

full_df, train_df, test_df, validation_df = prlib.get_preprocessed_data()

X_train, X_test, Y_train, Y_test = train_test_split(full_df.iloc[:, 0:-1], full_df['label'], stratify=full_df['label'], test_size=0.33, random_state=42)

#X_train, Y_train = df_without_label.iloc[:len(train_df), :], full_df['label'].iloc[:len(train_df)]
#X_test, Y_test = df_without_label.iloc[len(train_df):, :], full_df['label'].iloc[len(train_df):]

# PCA
X_train, embedding = prlib.get_PCs(X_train, 95)
X_test = embedding(X_test)

# MDS with mahalanobis
#df_mds = mds_mahalanobis(X_train, 70)
#df_mds['label'] = df['label']

def train_simple():
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

def train_svm(n_iter):
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

def train_rfc(n_iter):
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search_rf = RandomizedSearchCV(rfc, param_distributions=param_grid_rf, n_iter=n_iter, cv=5, verbose=3, random_state=42, n_jobs=-1, scoring = 'f1_macro')
    random_search_rf.fit(X_train, Y_train)
    print("Best parameters:", random_search_rf.best_params_)
    print("Best score:", random_search_rf.best_score_)
    return random_search_rf

if __name__ == '__main__':
    n_iter = int(sys.argv[1])
    if n_iter == 0:
        train_simple()
    else:
        train_svm(n_iter)
        #train_rfc(n_iter)