
from mido import MidiFile, Message, MetaMessage, MidiTrack
from math import ceil
import os
import numpy as np
from sklearn.utils import shuffle
from config import cfg

#Global parameters
time_per_time_slice = cfg.CONST.TIME_PER_TIME_SLICE #0.02 #200ms #time-unit for each column in the piano roll
highest_note = cfg.CONST.HIGHEST_NOTE #81 # A_6
lowest_note = cfg.CONST.LOWEST_NOTE  #33 # A_2
input_dim = cfg.CONST.INPUT_DIM #highest_note - lowest_note + 1
output_dim = cfg.CONST.OUTPUT_DIM #highest_note - lowest_note + 1
MICROSECONDS_PER_MINUTE = cfg.CONST.MICROSECONDS_PER_MINUTE #60000000


def midiToPianoroll(filepath, debug = False):
	midi_data = MidiFile(filepath)
	resolution = midi_data.ticks_per_beat
	if debug:
		print ("resolution", resolution)
	set_tempo_events = [x for t in midi_data.tracks for x in t if str(x.type) == 'set_tempo']
	
	tempo = MICROSECONDS_PER_MINUTE/set_tempo_events[0].tempo
	if debug:
		print ("tempo", tempo)
	ticks_per_time_slice = 1.0 * (resolution * tempo * time_per_time_slice)/60 
	if debug:
		print ("ticks_per_time_slice", ticks_per_time_slice)
	
	#Get maximum ticks across all tracks
	total_ticks =0
	for t in midi_data.tracks:
        #since ticks represent delta times we need a cumulative sum to get the total ticks in that track
		sum_ticks = 0
		for e in t:
			if str(e.type) == 'note_on' or str(e.type) == 'note_off' or str(e.type) == 'end_of_track':
				sum_ticks += e.time
				
		if sum_ticks > total_ticks:
			total_ticks = sum_ticks
	if debug:
		print ("total_ticks", total_ticks)

	time_slices = int(ceil(total_ticks / ticks_per_time_slice))
	if debug:
		print ("time_slices", time_slices)

	piano_roll = np.zeros((input_dim, time_slices), dtype =int)

	note_states = {}
	for track in midi_data.tracks:
		total_ticks = 0
		for event in track:
			if str(event.type) == 'note_on' and event.velocity > 0:
				total_ticks += event.time
				time_slice_idx = int(total_ticks / ticks_per_time_slice )

				if event.note <= highest_note and event.note >= lowest_note: 
					note_idx = event.note - lowest_note
					piano_roll[note_idx][time_slice_idx] = 1
					note_states[note_idx] = time_slice_idx

			elif str(event.type) == 'note_off' or ( str(event.type) == 'note_on' and event.velocity == 0 ):
				note_idx = event.note - lowest_note
				total_ticks += event.time
				time_slice_idx = int(total_ticks /ticks_per_time_slice )

				if note_idx in note_states:	
					last_time_slice_index = note_states[note_idx]
					piano_roll[note_idx][last_time_slice_index:time_slice_idx] = 1
					del note_states[note_idx]
	return piano_roll.T

#preprocess data directory
def get_data(data_dir):
	pianoroll_data = []
	for file in os.listdir(data_dir):
		filepath = data_dir + "/" + file
		piano_roll = midiToPianoroll(filepath)
		pianoroll_data.append(piano_roll)

	return pianoroll_data

def createSeqNetInputs(pianoroll_data, x_seq_length, y_seq_length):
	x = []
	y = []

	for i,piano_roll in enumerate(pianoroll_data):
		print ("piano_roll.shape", piano_roll.shape)
		
		pos = 0
		while pos + x_seq_length + y_seq_length < piano_roll.shape[0]:
			x.append(piano_roll[pos:pos + x_seq_length])
			y.append(piano_roll [pos+ x_seq_length: pos + x_seq_length + y_seq_length])
			pos += x_seq_length

	X = np.array(x)
	Y = np.array(y)
	print ("x shape", X.shape)
	print ("y shape", Y.shape)

	x_1, y_1 = shuffle(X,Y)

	return x_1, y_1

#create Network inputs
def createSeqTestNetInputs(pianoroll_data, seq_length):
	x_test = []
	
	for i,piano_roll in enumerate(pianoroll_data):
		print ("piano_roll.shape", piano_roll.shape)
		x = []
		pos = 0
		while pos + seq_length < piano_roll.shape[0]:
			x.append(piano_roll[pos:pos + seq_length])
			pos +=1
		x_test.append(np.array(x))

	print("x_test shape", np.array(x_test).shape)

	return np.array(x_test)

# NN output to pianoroll
def seqNetOutToPianoroll(output, threshold = 0.1):
	piano_roll = []
	for seq_out in output:
		for time_slice in seq_out:
			idx = [i for i,t in enumerate(time_slice) if t > threshold]
			pianoroll_slice = np.zeros(time_slice.shape)
			pianoroll_slice[idx] = 1
			piano_roll.append(pianoroll_slice)

	return np.array(piano_roll)
    
# pianoroll to MIDI
def pianorollToMidi(piano_roll, filepath):
    #ensure that resolution is an integer 
	ticks_per_time_slice=1 # hard-coded, arbitrary but needs to be >= 1 and an integer to avoid distortion
	tempo = 1/time_per_time_slice
	resolution = 60*ticks_per_time_slice/(tempo*time_per_time_slice)

	mid = MidiFile(ticks_per_beat = int(resolution))
	track = MidiTrack()
	mid.tracks.append(track)
	track.append(MetaMessage('set_tempo', tempo = int(MICROSECONDS_PER_MINUTE/tempo), time =0))

	current_state = np.zeros(input_dim)

	index_of_last_event = 0

	for slice_index, time_slice in enumerate(np.concatenate((piano_roll, np.zeros((1, input_dim))), axis =0)):
		note_changes = time_slice - current_state
		
		for note_idx, note in enumerate(note_changes):
			if note == 1:
				note_event = Message('note_on', time = (slice_index - index_of_last_event)*ticks_per_time_slice, velocity = 65, note = note_idx + lowest_note )
				track.append(note_event)
				index_of_last_event = slice_index
			elif note == -1:
				note_event = Message('note_off', time = (slice_index - index_of_last_event)*ticks_per_time_slice, velocity = 65, note = note_idx + lowest_note )
				track.append(note_event)
				index_of_last_event = slice_index

		current_state = time_slice

	eot = MetaMessage('end_of_track', time=1)
	track.append(eot)
	
	mid.save(filepath)

