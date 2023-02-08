'''
Author: Vitor Abdo

This is the main system file that runs all the necessary 
components to run the machine learning pipeline
'''

# import necessary packages
import mlflow
import tempfile
import os
import wandb
import hydra
import json
from omegaconf import DictConfig

_steps = [
    'upload_raw_data',
    'transform_raw_data',
    'basic_clean',
    'data_check',
    'train_model']

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    '''main file that runs the entire pipeline end-to-end using hydra and mlflow

    :param config: (.yaml file) 
    file that contains all the default data for the entire machine learning pipeline to run
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(',') if steps_par != 'all' else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if 'upload_raw_data' in active_steps:
            # Download file from source and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/01_upload_raw_data",
                'main',
                version='main',
                parameters={
                    'artifact_name': 'raw_data',
                    'artifact_type': 'dataset',
                    'artifact_description': 'Raw dataset used for the project, pulled directly from airbnb',
                    'input_uri': config['01_upload_raw_data']['input_uri']
                },
            )

        if 'transform_raw_data' in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/02_transform_raw_data",
                'main',
                version='main',
                parameters={
                    'input_artifact': config['02_transform_raw_data']['input_artifact'],
                    'test_size': config['02_transform_raw_data']['test_size'],
                    'random_seed': config['02_transform_raw_data']['random_seed'],
                    'stratify_by': config['02_transform_raw_data']['stratify_by'],
                    'artifact_description': 'Raw dataset transformed with some necessary functions and then divided between training and testing to start the data science pipeline'
                },
            )

        if 'basic_clean' in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/04_basic_clean",
                'main',
                version='main',
                parameters={
                    'input_artifact': config['04_basic_clean']['input_artifact'],
                    'artifact_name': 'clean_data',
                    'artifact_type': 'dataset',
                    'artifact_description': 'Clean dataset after we apply "clean_data" function',
                    'min_price': config['04_basic_clean']['min_price'],
                    'max_price': config['04_basic_clean']['max_price'],
                    'min_nights': config['04_basic_clean']['min_nights'],
                    'max_nights': config['04_basic_clean']['max_nights']
                },
            )

        if 'data_check' in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/05_data_check",
                'main',
                version='main',
                parameters={
                    'csv': config['05_data_check']['csv'],
                    'ref': config['05_data_check']['ref'],
                    'kl_threshold': config['05_data_check']['kl_threshold'],
                    'min_price': config['05_data_check']['min_price'],
                    'max_price': config['05_data_check']['max_price']
                },
            )

        if 'train_model' in active_steps:
             rf_config = os.path.abspath('rf_config.json')
             with open(rf_config, 'w+') as fp:
                 json.dump(dict(config['06_train_model']['random_forest'].items()), fp)
                 
             _ = mlflow.run(
                 f"{config['main']['components_repository']}/06_train_model",
                 'main',
                 version='main',
                 parameters={
                    'input_artifact': config['06_train_model']['input_artifact'],
                    'random_seed': config['06_train_model']['random_seed'],
                    'rf_config': rf_config,
                    'cv': config['06_train_model']['cv'],
                    'artifact_name': 'final_model_pipe',
                    'artifact_type': 'pickle',
                    'artifact_description': 'Final model pipeline after training, exported in the correct format for making inferences'
                },
            )
        

