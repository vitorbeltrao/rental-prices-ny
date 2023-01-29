'''
Author: Vitor Abdo

This .py file serves to download the first artifact 
uploaded from the last step and does some initial 
transformations at the etl level only 
(not at the ML model level) 
'''

# import necessary packages
import logging
import csv
import argparse
import wandb
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# main function of this file
def transform_raw_data(args) -> None:
    '''Function that receives the raw data coming from the data source 
    and applies some necessary transformations (etl).

    These transformations are just to get the data in the right shape 
    for us to start the data science pipeline.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='rental-prices-ny', entity='vitorabdo', job_type="transform_data")
    artifact = run.use_artifact(args.input_artifact, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # transform downloaded dataset
    df_raw = pd.read_csv(filepath, low_memory=False)
    df_transformed = df_raw.drop(
        ['license', 'id', 'name', 'host_id', 
        'host_name', 'last_review'], axis=1)
    logger.info("Transformed raw data: SUCCESS")

    # divide the dataset into train and test
    train_set, test_set = train_test_split(
        df_transformed, 
        test_size=args.test_size,
        random_state =args.random_seed,
        stratify=df_transformed[args.stratify_by] if args.stratify_by != 'none' else None)
    logger.info("Splitted transformed data into train and test: SUCCESS")

    # upload to W&B
    for df, name in zip([train_set, test_set], ['train_set', 'test_set']):
        logger.info(f"Uploading {name}.csv dataset")
        artifact = wandb.Artifact(
            name=name,
            type='dataset',
            description=args.artifact_description)

        temp_csv = df.to_csv(name + '.csv', index=False)
        print(temp_csv)
        artifact.add_file(temp_csv)
        run.log_artifact(artifact)
        logger.info(f"Uploaded {name}.csv: SUCCESS")


if __name__ == "__main__":
    logging.info('About to start executing the transform_raw_data function')

    parser = argparse.ArgumentParser(
        description='Upload an artifact to W&B. Adds a reference denoted by a csv to the artifact.')

    parser.add_argument(
        '--input_artifact',
        type=str,
        help='String referring to the W&B directory where the csv with the raw data to be transformed is located.',
        required=True)

    parser.add_argument(
        '--test_size', 
        type=float, 
        help='Size of the test split. Fraction of the dataset, or number of items',
        required=False,
        default=0.2)

    parser.add_argument(
        '--random_seed', 
        type=int, 
        help='Seed for random number generator', 
        required=False,
        default=42)

    parser.add_argument(
        '--stratify_by', 
        type=str, 
        help='Column to use for stratification', 
        required=False,
        default='none')

    parser.add_argument(
        '--artifact_description',
        type=str,
        help='Free text that offers a description of the artifact.',
        required=False,
        default='Raw dataset transformed with some necessary functions and then divided between training and testing to start the data science pipeline')

    args = parser.parse_args()
    transform_raw_data(args)
    logging.info('Done executing the transform_raw_data function')

