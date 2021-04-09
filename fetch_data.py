"""
Script for fetching ModelNet40 Dataset
"""

import os
import requests
import zipfile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def get_modelnet_40(data_dir='./data/', data_url='http://modelnet.cs.princeton.edu/ModelNet40.zip', extract_file=True):
    """
    Download ModelNet40 dataset

    Args:
    - data_dir: String. A directory where fetched data will be stored
    - data_src: String. URL of the dataset
    - extract_file: Bool. Extract zip file and remove it if true, do nothing otherwise
    """

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    filename = 'modelnet40.zip'

    file_path = "{}{}".format(data_dir, filename)
    
    if os.path.isfile(file_path):
        print('[!] File already exists. Fetching is not required.')
    else:
        # create progress bar
        response = requests.get(data_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        # download file, visualize progress
        with open(file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('[!] ERROR, something went wrong')
            exit(-1)

    if extract_file:
        # extract files
        print('[!] Extracting files...')
        zipped_file = zipfile.ZipFile(file_path)
        zipped_file.extractall('./data')
        zipped_file.close()

        # remove zipped file
        print('[!] Cleaning up...')
        os.remove(file_path)

if __name__ == '__main__':
    get_modelnet_40()
