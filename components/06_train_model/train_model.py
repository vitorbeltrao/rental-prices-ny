'''
Author: Vitor Abdo

This .py file is for training, saving the best model and
get the feature importance for model
'''

# Import necessary packages
import argparse
import wandb
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate
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


def feature_importance_plot(pipe: Pipeline, feat_names: list) -> plt.figure:
    '''Function to generate the graph of the
    most important variables for the model
    '''
    # we collect the feature importance for all non-nlp features first
    feat_imp = pipe['random_forest'].feature_importances_[: len(feat_names)-1]

    # plot
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color='r', align='center')
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp

def train_model(args):
    '''Function to train the model, tune the hyperparameters
    and save the best final model
    '''
    # start a new run at wandb
    run = wandb.init(
        project='rental-prices-ny',
        entity='vitorabdo',
        job_type='train_data')
    artifact = run.use_artifact(args.input_artifact, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded cleaned data artifact: SUCCESS')

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # select only the features that we are going to use
    df_clean = pd.read_csv(filepath)
    X = df_clean.drop(['price'], axis=1)
    y = df_clean['price']
    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # training the model
    logger.info('Preparing sklearn pipeline')
    sk_pipe, processed_features = get_inference_pipeline(rf_config)

    logger.info('Fitting...')
    scores = cross_validate(sk_pipe, X, y, return_train_score=True,
                            scoring=('r2', 'neg_mean_squared_error'), cv=args.cv)

    # compute r2 and RMSE
    logger.info('Scoring...')
    train_r2_scores = np.mean(scores['train_r2'])
    test_r2_scores = np.mean(scores['test_r2'])

    train_rmse_scores = np.mean(np.sqrt(-scores['train_neg_mean_squared_error']))
    test_rmse_scores = np.mean(np.sqrt(-scores['test_neg_mean_squared_error']))

    logger.info(f"Train_r2: {train_r2_scores}")
    logger.info(f"Test_r2: {test_r2_scores}")
    logger.info(f"Train_rmse: {train_rmse_scores}")
    logger.info(f"Test_rmse: {test_rmse_scores}")

    # exporting the model: save model package in the MLFlow sklearn format
    logger.info('Exporting model')
    
    mlflow.sklearn.save_model(
            sk_pipe,
            '06_train_model',
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    
    # upload the model artifact into wandb
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description)
    
    artifact.add_file('final_model_pipe.pkl')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')

    # Plot feature importance
    fig_feat_imp = feature_importance_plot(sk_pipe, processed_features)

    # lets save and upload all metrics to wandb
    run.summary['Train_r2'] = train_r2_scores
    run.summary['Test_r2'] = test_r2_scores
    run.summary['Train_rmse'] = train_rmse_scores
    run.summary['Test_rmse'] = test_rmse_scores

    run.log(
        {
          'feature_importance': wandb.Image(fig_feat_imp)
        }
    )


if __name__ == "__main__":
    logging.info('About to start executing the train_model function')

    parser = argparse.ArgumentParser(
        description='Upload an artifact to W&B. Adds a reference denoted by a pkl to the artifact.')

    parser.add_argument(
        '--input_artifact',
        type=str,
        help='String referring to the W&B directory where the csv with the cleaned dataset to be trained is located.',
        required=True)
    
    parser.add_argument(
        '--rf_config',
        type=str,
        help='Path to a YAML file containing the configuration for the random forest.',
        required=True)
    
    parser.add_argument(
        '--cv',
        type=int,
        help='The number of folds to apply in cross-validation.',
        required=False,
        default=5)

    parser.add_argument(
        '--artifact_name',
        type=str,
        help='A human-readable name for this artifact which is how you can identify this artifact.',
        required=True)

    parser.add_argument(
        '--artifact_type',
        type=str,
        help='The type of the artifact, which is used to organize and differentiate artifacts.',
        required=True)

    parser.add_argument(
        '--artifact_description',
        type=str,
        help='Free text that offers a description of the artifact.',
        required=False,
        default='Final model pipeline after training, exported in the correct format for making inferences.')

    args = parser.parse_args()
    train_model(args)
    logging.info('Done executing the train_model function')