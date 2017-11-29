import os
from easydict import EasyDict as edict
from keras.optimizers import Adam
import time

_C = edict()
cfg = _C

#DATA DIRECTORIES
_C.DATA =edict()
_C.DATA.BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

_C.DATA.WEIGHTS_FILE = 'LSTM_weights_20171116_21_48' #'LSTM_weights_%s' %(time.strftime("%Y%m%d_%H_%M"))
_C.DATA.WEIGHTS_DIR = os.path.join(cfg.DATA.BASE_FOLDER, "..", "weights")
_C.DATA.WEIGHTS_PATH = '%s%s' %(cfg.DATA.WEIGHTS_DIR, cfg.DATA.WEIGHTS_FILE)

_C.DATA.MODEL_FILE = 'LSTM_model_20171116_21_48' #'LSTM_model_%s' %(time.strftime("%Y%m%d_%H_%M"))
_C.DATA.MODEL_DIR = os.path.join(cfg.DATA.BASE_FOLDER, "..", "checkpoints")
_C.DATA.MODEL_PATH = '%s%s' %(cfg.DATA.MODEL_DIR, cfg.DATA.MODEL_FILE)

_C.DATA.GENERATED_DIR = os.path.join(cfg.DATA.BASE_FOLDER, "..", "generated")

#AZURE parameters
_C.AZURE = edict()
_C.AZURE.ACCOUNT_NAME = os.environ['STORAGE_ACCOUNT_NAME']
_C.AZURE.ACCOUNT_KEY = os.environ['STORAGE_ACCOUNT_KEY']

#Global paramters
_C.CONST = edict()
_C.CONST.TIME_PER_TIME_SLICE = 0.02 #200ms #time-unit for each column in the piano roll
_C.CONST.HIGHEST_NOTE = 81 # A_6
_C.CONST.LOWEST_NOTE = 33 # A_2
_C.CONST.INPUT_DIM = cfg.CONST.HIGHEST_NOTE - cfg.CONST.LOWEST_NOTE + 1
_C.CONST.OUTPUT_DIM = cfg.CONST.HIGHEST_NOTE - cfg.CONST.LOWEST_NOTE + 1
_C.CONST.MICROSECONDS_PER_MINUTE = 60000000

#Model parameters
_C.MODEL_PARAMS = edict()
_C.MODEL_PARAMS.NUM_UNITS = 64
_C.MODEL_PARAMS.X_SEQ_LENGTH = 50
_C.MODEL_PARAMS.Y_SEQ_LENGTH = 50
_C.MODEL_PARAMS.LOSS_FUNCTION = 'categorical_crossentropy'
_C.MODEL_PARAMS.OPTIMIZER = Adam() #lr=0.0001
_C.MODEL_PARAMS.BATCH_SIZE = 64
_C.MODEL_PARAMS.NUM_EPOCHS = 100