'''
Author: Vitor Abdo

This .py file is used to clean up the data,
for example removing outliers.
'''

# import necessary packages
import logging
import argparse
import wandb
import pandas as pd

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def clean_data(args) -> None:
    '''Function to clean up our training dataset to feed the machine
    learning model.
    '''
    # start a new run at wandb
    run = wandb.init(
        project='rental-prices-ny',
        entity='vitorabdo',
        job_type='clean_data')
    artifact = run.use_artifact(args.input_artifact, type='dataset')
    filepath = artifact.file()
    logger.info('Downloaded raw data artifact: SUCCESS')

    # clean the train dataset
    df_raw = pd.read_csv(filepath, low_memory=False)
    df_clean = df_raw.loc[
        (df_raw['price'] >= args.min_price) &
        (df_raw['price'] <= args.max_price) &
        (df_raw['minimum_nights'] >= args.min_nights) &
        (df_raw['minimum_nights'] <= args.max_nights)]
    logger.info('Train dataset are clean: SUCCESS')

    # upload to W&B
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description)

    df_clean.to_csv('df_clean.csv', index=False)
    artifact.add_file('df_clean.csv')
    run.log_artifact(artifact)
    logger.info('Artifact Uploaded: SUCCESS')


if __name__ == "__main__":
    logging.info('About to start executing the clean_data function')

    parser = argparse.ArgumentParser(
        description='Upload an artifact to W&B. Adds a reference denoted by a csv to the artifact.')

    parser.add_argument(
        '--input_artifact',
        type=str,
        help='String referring to the W&B directory where the csv with the train set to be transformed is located.',
        required=True)

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
        default='Clean dataset after we apply "clean_data" function')

    parser.add_argument(
        '--min_price',
        type=int,
        help='Minimum value you want to keep in the training dataset for the target variable.',
        required=False,
        default=10)

    parser.add_argument(
        '--max_price',
        type=int,
        help='Maximum value you want to keep in the training dataset for the target variable.',
        required=False,
        default=5944)

    parser.add_argument(
        '--min_nights',
        type=int,
        help='Minimum value you want to keep in the training dataset for the minimum_nights variable.',
        required=False,
        default=1)

    parser.add_argument(
        '--max_nights',
        type=int,
        help='Maximum value you want to keep in the training dataset for the minimum_nights variable.',
        required=False,
        default=370)

    args = parser.parse_args()
    clean_data(args)
    logging.info('Done executing the clean_data function')