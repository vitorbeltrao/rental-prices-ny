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

# basic logs config
logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# main function of this file
def upload_raw_data(args) -> csv:
    '''Function that upload an artifact, in this
    case a raw dataset for weights and biases
    '''
