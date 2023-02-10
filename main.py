'''
Author: Vitor Abdo

This is the main system file that runs all the necessary
components to run the machine learning pipeline
'''

# import necessary packages
import os
import json
import hydra
import mlflow
import tempfile
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def go(config: DictConfig):
    '''main file that runs the entire pipeline end-to-end using hydra and mlflow

    :param config: (.yaml file)
    file that contains all the default data for the entire machine learning pipeline to run
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ['WANDB_PROJECT'] = config['main']['project_name']
    os.environ['WANDB_RUN_GROUP'] = config['main']['experiment_name']

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    steps_to_execute = config["main"]["execute_steps"]
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if 'upload_raw_data' in steps_to_execute:
            # Download file from source and load in W&B
            _ = mlflow.run(
                os.path.join(
                root_path, "components", "01_upload_raw_data"), "main",
                parameters={
                    'artifact_name': 'raw_data',
                    'artifact_type': 'dataset',
                    'artifact_description': 'Raw dataset used for the project, pulled directly from airbnb',
                    'input_uri': config['01_upload_raw_data']['input_uri']
                },
            )

        if 'transform_raw_data' in steps_to_execute:
            _ = mlflow.run(
                os.path.join(
                root_path, "components", "02_transform_raw_data"), "main",
                parameters={
                    'input_artifact': config['02_transform_raw_data']['input_artifact'],
                    'test_size': config['02_transform_raw_data']['test_size'],
                    'random_seed': config['02_transform_raw_data']['random_seed'],
                    'stratify_by': config['02_transform_raw_data']['stratify_by'],
                    'artifact_description': 'Raw dataset transformed with some necessary functions and then divided between training and testing to start the data science pipeline'
                },
            )

        if 'basic_clean' in steps_to_execute:
            _ = mlflow.run(
                os.path.join(
                root_path, "components", "04_basic_clean"), "main",
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

        if 'data_check' in steps_to_execute:
            _ = mlflow.run(
                os.path.join(
                root_path, "components", "05_data_check"), "main",
                parameters={
                    'csv': config['05_data_check']['csv'],
                    'ref': config['05_data_check']['ref'],
                    'kl_threshold': config['05_data_check']['kl_threshold'],
                    'min_price': config['05_data_check']['min_price'],
                    'max_price': config['05_data_check']['max_price']
                },
            )

        if 'train_model' in steps_to_execute:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                fp.write(OmegaConf.to_yaml(config["06_train_model"]))

            _ = mlflow.run(
                os.path.join(root_path, "components", "06_train_model"), "main",
                parameters={
                    'input_artifact': config['06_train_model']['input_artifact'],
                    'rf_config': rf_config,
                    'cv': config['06_train_model']['cv'],
                    'scoring': config['06_train_model']['scoring'],
                    'artifact_name': 'final_model_pipe',
                    'artifact_type': 'pickle',
                    'artifact_description': 'Final model pipeline after training, exported in the correct format for making inferences'
                },
            )

        if 'test_model' in steps_to_execute:
            _ = mlflow.run(
                os.path.join(
                root_path, "components", "07_test_model"), "main",
                parameters={
                    'mlflow_model': config['07_test_model']['mlflow_model'],
                    'test_data': config['07_test_model']['test_data'],
                    'confidence_level': config['07_test_model']['confidence_level']
                },
            )


if __name__ == "__main__":
    go()
