'''
Author: Vitor Abdo

This .py file serves to download the first artifact
uploaded from the last step and split the raw
dataset into train and test set.
'''

# import necessary packages
import logging
import argparse
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def transform_raw_data(args) -> None:
    '''Function that receives the raw data coming from the data source
    and divide the raw data into train and test set.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='rental-prices-ny',
        entity='vitorabdo',
        job_type='transform_data')
    artifact = run.use_artifact(args.input_artifact, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # divide the dataset into train and test
    df_raw = pd.read_csv(filepath, low_memory=False)
    train_set, test_set = train_test_split(
        df_raw,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df_raw[args.stratify_by] if args.stratify_by != 'none' else None)
    logger.info('Splitted raw data into train and test: SUCCESS')

    # upload to W&B
    for df, name in zip([train_set, test_set], ['train_set', 'test_set']):
        logger.info(f'Uploading {name}.csv dataset')
        artifact = wandb.Artifact(
            name=name,
            type='dataset',
            description=args.artifact_description)

        df.to_csv(name + '.csv', index=False)
        artifact.add_file(name + '.csv')
        run.log_artifact(artifact)
        logger.info(f'Uploaded {name}.csv: SUCCESS')


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
        help='Size of the test split. Fraction of the dataset, or number of items.',
        required=False,
        default=0.2)

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Seed for random number generator.',
        required=False,
        default=42)

    parser.add_argument(
        '--stratify_by',
        type=str,
        help='Column to use for stratification.',
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
