'''
Author: Vitor Abdo

This .py file serves to upload the raw data 
in W&B extracted from the data source
'''

# import necessary packages
import logging
import wandb

# basic logs config
logging.basicConfig(
    filename='rental-prices-ny/logs/logs_system_funcs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# main function of this file
def upload_raw_data():
    '''
    '''


