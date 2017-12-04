from __future__ import print_function
import zipfile
import os
from sys import platform
import shutil
import glob
from config import cfg

try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
def download_grocery_data():
    
    dataset_folder = os.path.join(cfg.DATA.BASE_FOLDER, "..")
    if not os.path.exists(os.path.join(dataset_folder, "midi")):
        filename = os.path.join(dataset_folder, "scale_chords.zip")
        if not os.path.exists(filename):
            url = "http://www.feelyoursound.com/data/scale_chords_small.zip"
            print('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(dataset_folder)

            files = glob.glob(os.path.join(dataset_folder, "midi") + '/*.mid')
              
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder + '/data')

    return os.path.join(dataset_folder,"midi")
    
if __name__ == "__main__":
    download_grocery_data()