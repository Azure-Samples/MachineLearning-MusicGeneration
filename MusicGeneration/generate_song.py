# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path
from azure.storage.blob import BlockBlobService
from azureml.logging import get_azureml_logger

from download_data import download_grocery_data
from midi_io import midiToPianoroll, createSeqTestNetInputs, seqNetOutToPianoroll, pianorollToMidi
from keras.models import model_from_json
from config import cfg 

import time
import random
import glob
import numpy as np

block_blob_service = BlockBlobService(account_name= cfg.AZURE.ACCOUNT_NAME, account_key=cfg.AZURE.ACCOUNT_KEY)

def run(test_data):

    generated_file = 'LSTM_gen_%s.mid' %(time.strftime("%Y%m%d_%H_%M"))
    generated_path = '%s%s' %(cfg.DATA.GENERATED_DIR, generated_file)

    for i,song in enumerate(test_data):
        net_output = model.predict(song)
        print("net_output:", np.array(net_output.shape))
        net_roll = seqNetOutToPianoroll(net_output)
        print("net_roll:", net_roll.shape)
        pianorollToMidi(net_roll, generated_path)

    block_blob_service.create_blob_from_path('musicmodels', generated_file, generated_path)

def init():
    
    global model
    #load model and weights from blob storage
    block_blob_service.get_blob_to_path('musicmodels', cfg.DATA.MODEL_FILE, cfg.DATA.MODEL_PATH)
    block_blob_service.get_blob_to_path('musicmodels', cfg.DATA.WEIGHTS_FILE, cfg.DATA.WEIGHTS_PATH)

    model = model_from_json(open(cfg.DATA.MODEL_PATH).read())
    model.load_weights(cfg.DATA.WEIGHTS_PATH)
    model.compile(loss= cfg.MODEL_PARAMS.LOSS_FUNCTION , optimizer=cfg.MODEL_PARAMS.OPTIMIZER)

if __name__ == '__main__':
    # Import the logger only for Workbench runs
    logger = get_azureml_logger()

    init()

    #PRIMER
    dataset_folder = download_grocery_data()
    midi_files  = glob.glob(dataset_folder +'/*.mid')
    
    #choose a random file as a primer
    file_idx = random.randint(0,len(midi_files) - 1)
    primer = midi_files[file_idx]
    test_piano_roll = midiToPianoroll(primer)
    test_data = [test_piano_roll]
    test_input = createSeqTestNetInputs(test_data, cfg.MODEL_PARAMS.X_SEQ_LENGTH)
    
    run(test_input)
    