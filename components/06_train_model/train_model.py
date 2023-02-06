'''
Author: Vitor Abdo

This .py file is for training, saving the best model and
get the feature importance for model
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