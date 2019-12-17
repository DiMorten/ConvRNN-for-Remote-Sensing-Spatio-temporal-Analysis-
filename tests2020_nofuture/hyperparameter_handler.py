from utils import *

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse
import tensorflow as tf


import sys
import glob
import pickle
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from densnet import DenseNetFCN
from densnet_timedistributed import DenseNetFCNTimeDistributed

from metrics import fmeasure,categorical_accuracy
import deb

import time
import csv
parser = argparse.ArgumentParser(description='')
parser.add_argument('-rfs', '--recurrent_filters', dest='recurrent_filters',
					type=int, default=128)
parser.add_argument('-mp', '--model_params', dest='model_params',
					nargs='+', default=None)
parser.add_argument('-id', '--model_id', dest='model_id',
					type=str, default=None) # valid: dense, unet, atrous
a = vars(parser.parse_args())
for idx in range(len(a['model_params'])):
	a['model_params'][idx]=int(a['model_params'][idx])
print("a",a)

# hp: hyperparameter values list
# All networks: 0: recurrent filter amount
# BDenseConvLSTM: 1: growth rate
#			2: nb_dense_block: number of downsampling..?
#			3: growth_rate
#			4: nb_layers_per_block: convolutional layers in each block
# BUnetConvLSTM:
# BAtrousConvLSTM 

#recurrent_filter_amount=256 No. Load from txt

def model_params_init(recurrent_filters=128):
	model_params={'dense':{'recurrent_filters':recurrent_filters,
				'nb_dense_block':2, # number of downsampling operations
				'growth_rate':64,
				'nb_layers_per_block':2}, # number of layers per block
		'unet':{'recurrent_filters':recurrent_filters,
				'filter_size':16},
		'atrous':{'recurrent_filters':recurrent_filters,
				'filter_size':16,
				'dilation_rate_mode':'auto', 
				'dilation_rates':[1,2,4,8]}
		}
	
	# Use only for manual. Otherwise not considered
	# auto / manual
	#w = csv.writer(open("model_parameters.csv", "w"))

	save_obj(model_params,'model_params')
def save_obj(obj, name ):
	with open('obj/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
def hyperparameter_fill(model_params,a):
	#print("a['model_params']",a['model_params'])
	if a['model_id']==None:
		print("Forgot to put model id")
		return 
	if a['model_params']==None or a['model_params']=="None":
		print("Forgot to put model params")
		return
	if len(a['model_params'])!=len(model_params[a['model_id']]): 
		print("Incorrect number of specified params")
		return
	count=0
	for key, val in model_params[a['model_id']].items():
		#print("key,val,a[key]",key,val,a['model_params'][count])
		#print()

		model_params[a['model_id']][key]=a['model_params'][count] # copy specified params to persistent var
		count+=1
	print(model_params[a['model_id']])
	save_obj(model_params,'model_params')	


if __name__ == '__main__':

	#mode='init'
	mode='write'
	if mode=='init':
		model_params_init()
	elif mode=='write':
		model_params=load_obj('model_params')
		hyperparameter_fill(model_params,a)
	print("End hyperparameter handler")
#hyperparameter_fill(a)

