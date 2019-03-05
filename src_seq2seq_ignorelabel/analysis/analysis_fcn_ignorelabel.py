 
import numpy as np

import cv2
import h5py
import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
from sklearn.externals import joblib

#===== normy3
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/fcn/seq1/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy/fcn/seq2/'

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_16/'

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_8/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_8/'


#======== normy3_check

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check/fcn/seq1/'
#=========== normy3_check2
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy3_check2/fcn/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq2/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq1/'

# gold

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3_check2/fcn/seq2/gold/'
# ====== normy3B

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3B/fcn/seq1/'
#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/normy3B/fcn/seq2/'


path='/home/lvc/Jorg/sbsr/fcn_model/keras_time_semantic_fcn/'
# ======= convlstm playground

path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/convlstm_playground/fcn_original500/'
path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/convlstm_playground/fcn_original/'

# === tranfer

path='/home/lvc/Jorg/igarss/fcn_transfer_learning_for_RS/results/transfer_fcn_seq2_to_seq1/'
# === normy3 check

path='/home/lvc/Jorg/igarss/fcn_transfer_learning_for_RS/results/normy3_check/seq1/fcn/'



# ======== ConvRNN

path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/densenet/'
prediction_path=path+'prediction.npy'

#prediction_path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/prediction_ConvLSTM_DenseNet_eyesight.npy'

# =========seq2seq 

path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/cv/'
prediction_path=path+'prediction_FCN_ConvLSTM_seq2seq_bi_skip_jokeillusion.npy'

prediction_path=path+'prediction_ConvLSTM_seq2seq_bi_whydyoucall.npy'

predictions=np.load(prediction_path)
label_test=np.load(path+'labels.npy')

# ================= Estimate the last timestamp
only_one_timestamp=False
if only_one_timestamp:
	label_test=label_test[:,-1,:,:,:]
	predictions=predictions[:,-1,:,:,:]


# ===== Oficial

#path='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/normy/fcn_8/'

print(predictions.shape)

print(label_test.shape)
#predictions=np.delete(predictions,range(2000,3000),axis=0)

class_n=predictions.shape[-1]

mode='each_date'

if mode=='each_date':
	predictions=predictions.argmax(axis=np.ndim(predictions)-1)
	label_test=label_test.argmax(axis=np.ndim(label_test)-1)

	for t in label_test.shape[1]:
		predictions_t=predictions[:,t,:,:]
		label_test_t=label_test[:,t,:,:]
		
		predictions_t=np.reshape(predictions_t,-1)
		label_test_t=np.reshape(label_test_t,-1)


	predictions=np.reshape(predictions,(predictions.shape[0]))
	

elif mode=='global':



	predictions=predictions.argmax(axis=np.ndim(predictions)-1)
	predictions=np.reshape(predictions,-1)
	#if mode=='best':
	label_test=label_test.argmax(axis=np.ndim(label_test)-1)
	label_test=np.reshape(label_test,-1)
	predictions=predictions[label_test<class_n]

	label_test=label_test[label_test<class_n]

	print("Predictions",predictions.shape)
	print("Label_test",label_test.shape)





	print(np.unique(predictions,return_counts=True))
	print(np.unique(label_test,return_counts=True))

	metrics=metrics_get(label_test,predictions)

#====================================

def metrics_get(label_test,predictions):
	print(predictions.shape,predictions.dtype)
	print(label_test.shape,label_test.dtype)



	metrics={}
	metrics['f1_score']=f1_score(label_test,predictions,average='macro')
	metrics['f1_score_weighted']=f1_score(label_test,predictions,average='weighted')
	        
	metrics['overall_acc']=accuracy_score(label_test,predictions)
	confusion_matrix_=confusion_matrix(label_test,predictions)
	metrics['per_class_acc']=(confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]).diagonal()
	print("acc",metrics['per_class_acc'])

	print(confusion_matrix_.sum(axis=1)[:, np.newaxis].diagonal())
	print(confusion_matrix_.diagonal())
	print(np.sum(confusion_matrix_,axis=1))
	acc=confusion_matrix_.diagonal()/np.sum(confusion_matrix_,axis=1)
	acc=acc[~np.isnan(acc)]
	print("Acc",acc)
	print("AA",np.average(acc))
	print("OA",np.sum(confusion_matrix_.diagonal())/np.sum(confusion_matrix_))


	metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])

	metrics['recall']=recall_score(label_test,predictions,average=None)
	metrics['precision']=precision_score(label_test,predictions,average=None)

	print(metrics)
	print(confusion_matrix_)

	print(metrics['precision'])
	print(metrics['recall'])
	return metrics



#metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])]


