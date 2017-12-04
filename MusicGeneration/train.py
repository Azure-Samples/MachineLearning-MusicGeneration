# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import *
from keras.layers.normalization import *
from keras.callbacks import EarlyStopping, History
from keras.layers import TimeDistributed
from keras.models import model_from_json
import time
from download_data import download_grocery_data
from midi_io import get_data, createSeqNetInputs
from config import cfg 
import sys
import os

from azureml.logging import get_azureml_logger
from azure.storage.blob import BlockBlobService

try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Library matplotlib missing. Can't plot")

block_blob_service = BlockBlobService(account_name= cfg.AZURE.ACCOUNT_NAME, account_key=cfg.AZURE.ACCOUNT_KEY)
from azure.storage.blob import PublicAccess
block_blob_service.create_container('musicmodels', public_access=PublicAccess.Container)

#Global parameters
time_per_time_slice = cfg.CONST.TIME_PER_TIME_SLICE #0.02 #200ms #time-unit for each column in the piano roll
highest_note = cfg.CONST.HIGHEST_NOTE #81 # A_6
lowest_note = cfg.CONST.LOWEST_NOTE  #33 # A_2
input_dim = cfg.CONST.INPUT_DIM #highest_note - lowest_note + 1
output_dim = cfg.CONST.OUTPUT_DIM #highest_note - lowest_note + 1
MICROSECONDS_PER_MINUTE = cfg.CONST.MICROSECONDS_PER_MINUTE #60000000

#Model parameters
num_units = cfg.MODEL_PARAMS.NUM_UNITS #64
x_seq_length = cfg.MODEL_PARAMS.X_SEQ_LENGTH #50
y_seq_length = cfg.MODEL_PARAMS.Y_SEQ_LENGTH  #50
loss_function = cfg.MODEL_PARAMS.LOSS_FUNCTION #'categorical_crossentropy'
optimizer = cfg.MODEL_PARAMS.OPTIMIZER #Adam() #lr=0.0001
batch_size = cfg.MODEL_PARAMS.BATCH_SIZE #64
num_epochs = cfg.MODEL_PARAMS.NUM_EPOCHS #100

# initialize the logger
logger = get_azureml_logger()
# This is how you log scalar metrics
logger.log("X_Seq_length", x_seq_length )
logger.log("y_Seq_length", y_seq_length )
logger.log("Loss Function", loss_function )
logger.log("Batch Size", batch_size )
logger.log("No Epochs", num_epochs )


def createSeq2Seq():
	#seq2seq model

	#encoder
	model = Sequential()
	model.add(LSTM(input_dim = input_dim, output_dim = num_units, activation= 'tanh', return_sequences = True ))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(LSTM(num_units, activation= 'tanh'))

	#decoder
	model.add(RepeatVector(y_seq_length))
	num_layers= 2
	for _ in range(num_layers):
		model.add(LSTM(num_units, activation= 'tanh', return_sequences = True))
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

	model.add(TimeDistributed(Dense(output_dim, activation= 'softmax')))
	return model


#Prepare data
dataset_folder = download_grocery_data()
pianoroll_data = get_data(dataset_folder)
input_data, target_data = createSeqNetInputs(pianoroll_data, x_seq_length, y_seq_length)
input_data = input_data.astype(np.bool)
target_data = target_data.astype(np.bool)

#Model
model = createSeq2Seq()
model.summary()
model.compile(loss=loss_function, optimizer = optimizer)
earlystop = EarlyStopping(monitor='loss', patience= 10, min_delta = 0.01 , verbose=0, mode= 'auto') 
history = History()
hist = model.fit(input_data, target_data, batch_size =  batch_size, nb_epoch=num_epochs, callbacks=[ earlystop, history ])
#print("History:", hist.history )

#Save model and weights to Blob storage
weights_file = 'LSTM_weights_%s' %(time.strftime("%Y%m%d_%H_%M"))
weights_path = '%s/%s' %(cfg.DATA.WEIGHTS_DIR, weights_file)
model.save_weights(weights_path)
print ("Weights saved to: ", weights_path)
block_blob_service.create_blob_from_path('musicmodels', weights_file, weights_path)


model_file = 'LSTM_model_%s' %(time.strftime("%Y%m%d_%H_%M"))
model_path = '%s/%s' %(cfg.DATA.MODEL_DIR, model_file)
json_string= model.to_json()
open(model_path, 'w').write(json_string)
print ("Model saved to: ", model_path)
block_blob_service.create_blob_from_path('musicmodels', model_file, model_path)


# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

fig = plt.figure(figsize=(6, 5), dpi=75)
plt.plot(hist.history['loss'])
fig.savefig("./outputs/Loss.png", bbox_inches='tight')

