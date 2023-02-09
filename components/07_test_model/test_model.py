'''
Author: Vitor Abdo

This file is for testing the final model with the "prod" tag in the test data
'''

# Import necessary packages
import argparse
import logging
import wandb
import mlflow
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def test_model(args):
    '''Function to test the model listed for production on the test dataset'''
    # start a new run at wandb
    run = wandb.init(
        project='rental-prices-ny',
        entity='vitorabdo',
        job_type='test_model')

    # download mlflow model
    model_local_path = run.use_artifact(
        args.mlflow_model, type='pickle').download()
    logger.info('Downloaded prod mlflow model: SUCCESS')

    # download test dataset
    test_data = run.use_artifact(args.test_data).file()
    logger.info('Downloaded test dataset artifact: SUCCESS')

    # Read test dataset
    test_data = pd.read_csv(test_data, low_memory=False)
    X_test = test_data.drop(['price'], axis=1)
    y_test = test_data['price']

    # making inference on test set
    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    # scoring the results
    logger.info("Scoring...")
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # confidence interval for the generalization error
    confidence = args.confidence_level
    squared_errors = (y_pred - y_test) ** 2
    confidence_interval = np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors)))

    logger.info(f"Test_r2: {r_squared}")
    logger.info(f"Test_rmse: {rmse}")
    logger.info(f"Test_confidence_interval: {confidence_interval}")

    # lets save and upload all metrics to wandb
    run.summary['Test_r2'] = r_squared
    run.summary['Test_rmse'] = rmse
    run.summary['Test_confidence_interval'] = confidence_interval


if __name__ == "__main__":
    logging.info('About to start executing the test_model function')

    parser = argparse.ArgumentParser(
        description='Test the provided model against the test dataset.')

    parser.add_argument(
        '--mlflow_model',
        type=str,
        help='String referring to the W&B directory where the mlflow production model is located.',
        required=True)

    parser.add_argument(
        '--test_data',
        type=str,
        help='String referring to the W&B directory where the csv with the test dataset to be tested is located.',
        required=True)

    parser.add_argument(
        '--confidence_level',
        type=float,
        help='Level of confidence you want in your RMSE assessment results.',
        required=False,
        default=0.95)

    args = parser.parse_args()
    test_model(args)
    logging.info('Done executing the test_model function')
