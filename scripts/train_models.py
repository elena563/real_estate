import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import sys
import pickle
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = os.path.join('../data', 'Real_estate_valuation_data_set.xlsx')
logging.info("Loading data")
df = pd.read_excel(DATA_PATH)
df = df.drop(columns=['No', 'X1 transaction date'])


# Feature extraction
X1 = df[['X5 latitude', 'X6 longitude']]
X2 = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
y = df['Y house price of unit area']

# Train-test split
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

X1_train, X1_test = X1.loc[train_idx], X1.loc[test_idx]
X2_train, X2_test = X2.loc[train_idx], X2.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

logging.info("Training models")
lr1 = LinearRegression()
lr1.fit(X1_train, y_train)
y_pred1 = lr1.predict(X1_test)

lr2 = LinearRegression()
lr2.fit(X2_train, y_train)
y_pred2 = lr2.predict(X2_test)

MODELS_PATH = 'linear_regression.pickle'
logging.info('Saving models...')
with open(MODELS_PATH, "wb") as file:   
    pickle.dump(lr1, file)
    pickle.dump(lr2, file)

# Create a DataFrame for the test set with predictions
test_df1 = df.loc[X1_test.index].copy()
test_df1['predictionlr'] = y_pred1 

test_df2 = df.loc[X2_test.index].copy()
test_df2['predictionlr'] = y_pred2