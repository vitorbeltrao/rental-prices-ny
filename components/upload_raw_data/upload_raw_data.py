'''
Author: Vitor Abdo

This .py file serves to upload the raw data 
in W&B extracted from the data source
'''

# import necessary packages
import logging
import argparse
import wandb

# basic logs config
logging.basicConfig(
    filename='rental-prices-ny/logs/logs_system_funcs.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# main function of this file
def upload_raw_data(args) -> None:
    '''Function that upload an artifact, in this 
    case a raw dataset for weights and biases
    '''
    # start a new run at wandb
    run = wandb.init(
        project="rental-prices-ny", entity="vitorabdo", job_type='upload_file')
    logger.info('Creating run for airbnb rental prices ny: SUCCESS')

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_reference(args.input_uri) # add a HTTP link to the artifact
    run.log_artifact(artifact) # save the artifact version to W&B and mark it as the output of this run
    logger.info('artifact uploaded to the wandb: SUCCESS')
    return None

if __name__ == "__main__":
    logging.info('About to start executing the upload_raw_data function')

    parser = argparse.ArgumentParser(
    description='Upload an artifact to W&B. Adds a reference denoted by a URI to the artifact.'
    )

    parser.add_argument(
        "--artifact_name", 
        type=str, 
        help='A human-readable name for this artifact, which is how you can identify this artifact.', 
        required=True
    )

    parser.add_argument(
        '--artifact_type', 
        type=str, 
        help='The type of the artifact, which is used to organize and differentiate artifacts.', 
        required=True
    )

    parser.add_argument(
        '--artifact_description',
        type=str,
        help='Free text that offers a description of the artifact.',
        required=False,
        default='Raw dataset used for the project, pulled directly from airbnb'
    )

    parser.add_argument(
        '--input_uri', 
        type=str, 
        help='Reference denoted by a URI (HTTP, for example) to the artifact.', 
        required=True
    )

    args = parser.parse_args()
    upload_raw_data(args)
    logging.info('Done executing the upload_raw_data function')


