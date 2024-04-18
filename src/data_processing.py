import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def clean(df):
    df = df.drop(columns=['index', 'Patient Id'])
    df = df.rename(columns={'Level': 'cancer level'})

    mapping = {
        'Low': 1,
        'Medium': 2,
        'High': 3
    }
    df['cancer level'] = df['cancer level'].map(mapping)

    return df

def feature_engineering(df):
    return df

def pca(df):
    # Get y column
    y = df['cancer level'].values

    # Get X column
    features = df.columns[:-1]  
    X = df.loc[:, features].values
    
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=None)
    pca.fit_transform(X_scaled)

    # Get eigenvalues (explained variance) of each component
    eigenvalues = pca.explained_variance_

    # Apply Kaiser Criterion: retain components with eigenvalues > 1
    kaiser_criteria_indices = np.where(eigenvalues > 1)[0]
    n_components_kaiser = len(kaiser_criteria_indices)
    print(f"Number of components to retain based on Kaiser Criterion: {n_components_kaiser}")
    
    pca_kaiser = PCA(n_components=n_components_kaiser)
    X = pca_kaiser.fit_transform(X_scaled)

    return X, y

def original_df(df):
    ## Cleaning ##
    df = clean(df)

    ## Feature engineering ##
    df = feature_engineering(df)

    ## Scale ##
    X, y = pca(df)

    return X, y