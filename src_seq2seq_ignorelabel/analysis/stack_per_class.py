 
import numpy as np

#import cv2
#import h5py
#import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
#====================================

dataset='cv'
if dataset=='cv':
	all_filenames=['averaged_metrics_cv_prediction_ConvLSTM_seq2seq_batch16_full.npy.csv',
	'averaged_metrics_cv_prediction_ConvLSTM_seq2seq_bi_batch16_full.npy.csv',
	'averaged_metrics_cv_prediction_DenseNetTimeDistributed_128x2_batch16_full.npy.csv',
	'averaged_metrics_cv_prediction_BAtrousGAPConvLSTM_raulapproved.npy.csv']
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv( "combined_csv_"+dataset+".csv", index=False)