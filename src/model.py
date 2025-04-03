from src import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import pickle
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import logging


def load_data():
    """Loads data from excel."""
    df = pd.read_excel(os.path.join(config.RAW_DATA_PATH, 'Real_estate_valuation_data_set.xlsx'))

    df = df.drop(columns=['No', 'X1 transaction date'])
    return df


def training():
    """Trains a Linear Regression and saves evaluation metrics to CSV."""
    df = load_data()

    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    X = df[['X5 latitude', 'X6 longitude']]
    # X = [df['X2 house age'], df['X3 distance to the nearest MRT station'], df['X4 number of convenience stores']]
    y = df['Y house price of unit area']

    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )
 
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    logging.info('Saving model...')
    with open(os.path.join(config.MODELS_PATH, "linear_regression.pickle"), "wb") as file:   
        pickle.dump(lr, file)

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['predictionlr'] = y_pred  # Add predictions


    '''# Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }'''

