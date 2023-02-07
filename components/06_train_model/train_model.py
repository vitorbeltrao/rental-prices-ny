'''
Author: Vitor Abdo

This .py file is for training, saving the best model and
get the feature importance for model
'''

# Import necessary packages
import wandb
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.compose import ColumnTransformer

import category_encoders as ce

from sklearn.impute import SimpleImputer

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_inference_pipeline(rf_config):
    '''function that creates the entire inference pipeline'''
    # preprocessing step
    # categorical values
    ordinal_categorical = ['room_type']
    non_ordinal_categorical = ['neighbourhood_group']

    # numerical values
    zero_imputed = [
        'minimum_nights',
        'number_of_reviews',
        'reviews_per_month',
        'calculated_host_listings_count',
        'availability_365']

    # categorical preprocessing
    ordinal_categorical_preproc = ce.OrdinalEncoder(
        cols = ordinal_categorical, 
        mapping = [
            {'col':'room_type',
            'mapping':{'Shared room':0,
                        'Private room':1,
                        'Entire home/apt':2,
                        'Hotel room':3}}])
        
    non_ordinal_categorical_preproc = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(drop='first'))

    # numerical preprocessing
    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)

    # apply the respective transformations with columntransformer method
    preprocessor = ColumnTransformer([
        ('ordinal_cat', ordinal_categorical_preproc, ordinal_categorical),
        ('non_ordinal_cat', non_ordinal_categorical_preproc, non_ordinal_categorical),
        ('impute_zero', zero_imputer, zero_imputed)],
        remainder='drop')
    
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed

    # instantiate the final model
    final_model = Pipeline(
        steps=[
            ('preprocessor', preprocessor), 
            ('scaling', StandardScaler()), 
            ('rf', RandomForestRegressor(**rf_config))
        ]
    )
    return final_model, processed_features


