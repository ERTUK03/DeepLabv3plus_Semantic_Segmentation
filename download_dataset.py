import requests
import os
import tarfile

def download_dataset(url, path):
    if os.path.isfile(f'{path}.tar'):
        print('File already exists. Skipping download.')
    else:
        response = requests.get(url)
        with open(f'{path}.tar', 'wb') as file:
            file.write(response.content)

    if os.path.isdir(path):
        print('Directory already exists. Skipping extraction.')
    else:
        with tarfile.open(f'{path}.tar', 'r') as tar_ref:
            tar_ref.extractall(path)
