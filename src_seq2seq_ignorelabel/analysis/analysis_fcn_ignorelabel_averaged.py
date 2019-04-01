 
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
def labels_predictions_filter_transform(label_test,predictions,class_n,
		debug=1):
	predictions=predictions.argmax(axis=np.ndim(predictions)-1)
	predictions=np.reshape(predictions,-1)
	label_test=label_test.argmax(axis=np.ndim(label_test)-1)
	label_test=np.reshape(label_test,-1)
	predictions=predictions[label_test<class_n]

	label_test=label_test[label_test<class_n]
	if debug>0:
		print("Predictions",predictions.shape)
		print("Label_test",label_test.shape)
	return label_test,predictions
def metrics_get(label_test,predictions,only_basics=False,debug=1):
	if debug>0:
		print(predictions.shape,predictions.dtype)
		print(label_test.shape,label_test.dtype)

	metrics={}
	metrics['f1_score']=f1_score(label_test,predictions,average='macro')
	metrics['overall_acc']=accuracy_score(label_test,predictions)
	confusion_matrix_=confusion_matrix(label_test,predictions)
	metrics['per_class_acc']=(confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]).diagonal()
	acc=confusion_matrix_.diagonal()/np.sum(confusion_matrix_,axis=1)
	acc=acc[~np.isnan(acc)]
	metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])
	if debug>0:
		print("acc",metrics['per_class_acc'])
		print("Acc",acc)
		print("AA",np.average(acc))
		print("OA",np.sum(confusion_matrix_.diagonal())/np.sum(confusion_matrix_))
		print("AA",metrics['average_acc'])
		print("OA",metrics['overall_acc'])

	if only_basics==False:

		metrics['f1_score_weighted']=f1_score(label_test,predictions,average='weighted')
		        

		metrics['recall']=recall_score(label_test,predictions,average=None)
		metrics['precision']=precision_score(label_test,predictions,average=None)
		if debug>0:
			print(confusion_matrix_.sum(axis=1)[:, np.newaxis].diagonal())
			print(confusion_matrix_.diagonal())
			print(np.sum(confusion_matrix_,axis=1))

			print(metrics)
			print(confusion_matrix_)

			print(metrics['precision'])
			print(metrics['recall'])
	return metrics


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
def experiment_analyze(dataset='cv',
		prediction_filename='prediction_DenseNetTimeDistributed_blockgoer.npy',
		mode='each_date',debug=1):
	path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'+dataset+'/'

	prediction_path=path+prediction_filename
	predictions=np.load(prediction_path)
	label_test=np.load(path+'labels.npy')
	if debug>0:
		print(predictions.shape)
		print(label_test.shape)
	class_n=predictions.shape[-1]

	if mode=='each_date':
		metrics_t={'f1_score':[],'overall_acc':[],
			'average_acc':[]}
		for t in range(label_test.shape[1]):
			predictions_t = predictions[:,t,:,:,:]
			label_test_t = label_test[:,t,:,:,:]

			label_test_t,predictions_t = labels_predictions_filter_transform(
				label_test_t, predictions_t, class_n=class_n,
				debug=debug)
			metrics = metrics_get(label_test_t, predictions_t,
				only_basics=True, debug=debug)	
			metrics_t['f1_score'].append(metrics['f1_score'])
			metrics_t['overall_acc'].append(metrics['overall_acc'])
			metrics_t['average_acc'].append(metrics['average_acc'])

		print(metrics_t)
		return metrics_t
	elif mode=='global':
		
		label_test,predictions=labels_predictions_filter_transform(
			label_test,predictions, class_n=class_n)

		print(np.unique(predictions,return_counts=True))
		print(np.unique(label_test,return_counts=True))

		metrics=metrics_get(label_test,predictions)

		return metrics

def experiments_analyze(dataset,experiment_list,mode='each_date'):
	experiment_metrics=[]
	for experiment in experiment_list:
		print("Starting experiment:",experiment)
		experiment_metrics.append(experiment_analyze(
			dataset=dataset,
			prediction_filename=experiment,
			mode=mode,debug=0))
	return experiment_metrics

def experiment_groups_analyze(dataset,experiment_group,mode='each_date'):
	save=True
	if save==True:	

		experiment_metrics=[]
		for group in experiment_group:
			group_metrics=[]
			for experiment in group:
				print("Starting experiment:",experiment)
				group_metrics.append(experiment_analyze(
					dataset=dataset,
					prediction_filename=experiment,
					mode=mode,debug=0))
			experiment_metrics.append(group_metrics)

	#	for model_id in range(len(experiment_metrics[0])):
	#		for date_id in range(len(experiment_metrics[0][model_id]):

		np.save('experiment_metrics.npy',experiment_metrics)
	else:
		experiment_metrics=np.load('experiment_metrics.npy')
	metrics={}
	total_metrics=[]

	for exp_id in range(len(experiment_metrics[0])):
		exp_id=int(exp_id)
		print(len(experiment_metrics))
		print(len(experiment_metrics[0]))
		print(experiment_metrics[0][0])
		print(experiment_metrics[0][0]['f1_score'])
		print(experiment_metrics[1][0]['f1_score'])

		print("exp_id",exp_id)		
		metrics['f1_score']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]),
			axis=0)
		print("metrics f1 score",metrics['f1_score'])
		metrics['overall_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['overall_acc'] for i in range(len(experiment_metrics))]),
			axis=0)

		metrics['average_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['average_acc'] for i in range(len(experiment_metrics))]),
			axis=0)
		total_metrics.append(metrics.copy())
		print("total metrics f1 score",total_metrics)

	print("metrics['f1_score'].shape",metrics['f1_score'].shape)
	print("total merics len",len(total_metrics))
	print(total_metrics)
	return total_metrics

def experiments_plot(metrics,experiment_list,dataset):



	if dataset=='cv':
		valid_dates=[0,2,4,5,6,8,10,11,13]
		t_len=len(valid_dates)
	else:
		t_len=len(metrics[0]['f1_score'])

	print("t_len",t_len)
	indices = range(t_len) # t_len
	X = np.arange(t_len)

	exp_id=0
	width=0.5
	colors=['b','y','c','m','r']
	#colors=['#7A9AAF','#293C4B','#FF8700']
	#colors=['#4225AC','#1DBBB9','#FBFA17']
	##colors=['b','#FBFA17','c']
	#colors=['#966A51','#202B3F','#DA5534']
	exp_handler=[] # here I save the plot for legend later
	exp_handler2=[] # here I save the plot for legend later
	exp_handler3=[] # here I save the plot for legend later

	figsize=(8,4)
	fig, ax = plt.subplots(figsize=figsize)
	fig2, ax2 = plt.subplots(figsize=figsize)
	fig3, ax3 = plt.subplots(figsize=figsize)

	fig.subplots_adjust(bottom=0.2)
	fig2.subplots_adjust(bottom=0.2)
	fig3.subplots_adjust(bottom=0.2)

	for experiment in experiment_list:
		print("experiment",experiment)
		print(exp_id)
		metrics[exp_id]['f1_score']=np.transpose(np.asarray(metrics[exp_id]['f1_score']))*100
		metrics[exp_id]['overall_acc']=np.transpose(np.asarray(metrics[exp_id]['overall_acc']))*100
		metrics[exp_id]['average_acc']=np.transpose(np.asarray(metrics[exp_id]['average_acc']))*100

		if dataset=='cv':
			
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
			metrics[exp_id]['f1_score']=metrics[exp_id]['f1_score'][valid_dates]
			metrics[exp_id]['overall_acc']=metrics[exp_id]['overall_acc'][valid_dates]
			metrics[exp_id]['average_acc']=metrics[exp_id]['average_acc'][valid_dates]
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
		exp_handler.append(ax.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['f1_score'], 
			color = colors[exp_id], width = width/2))
		ax.set_title('Average F1 Score (%)')
		ax.set_xlabel('Month')
		if dataset=='lm':
			xlim=[-0.5,13] 
			ylim=[10,76]
			xticklabels=['Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			ax.set_ylim(10,76)
			ax3.set_ylim(34,100)

			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)
			
		elif dataset=='cv': 
			xlim=[-0.3,8.9]
			xticklabels=['Oct','Nov','Dec','Jan','Feb','Mar','May','Jun','Jul']

			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			ax.set_ylim(5,73)
			ax3.set_ylim(57,94)

			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)

		exp_handler2.append(ax2.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['average_acc'], 
			color = colors[exp_id], width = width/2))
		ax2.set_title('Average Accuracy')
		ax2.set_xlabel('Month')
		exp_handler3.append(ax3.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['overall_acc'], 
			color = colors[exp_id], width = width/2))
		ax3.set_title('Overall Accuracy (%)')
		ax3.set_xlabel('Month')

		#ax3.set_xticks(np.arange(5))
		#ax3.set_xticklabels(('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
		
		exp_id+=1
	
	ax.legend(tuple(exp_handler), ('UConvLSTM','BConvLSTM','BDenseConvLSTM'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)
	ax2.legend(tuple(exp_handler2), ('UConvLSTM','BConvLSTM','BDenseConvLSTM'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)
	ax3.legend(tuple(exp_handler3), ('UConvLSTM','BConvLSTM','BDenseConvLSTM'),loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=3)

	#ax.set_rasterized(True)
	#ax2.set_rasterized(True)
	#ax3.set_rasterized(True)
	
	#fig.savefig("f1_score_"+dataset+".eps",format="eps",dpi=300)
	#fig2.savefig("average_acc_"+dataset+".eps",format="eps",dpi=300)
	#fig3.savefig("overall_acc_"+dataset+".eps",format="eps",dpi=300)
	fig.savefig("f1_score_"+dataset+".png",dpi=300)
	fig2.savefig("average_acc_"+dataset+".png",dpi=300)
	fig3.savefig("overall_acc_"+dataset+".png",dpi=300)

	#plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
	#plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
	#plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)


	#plot_metric=[x['f1_score'] for x in metrics]
	#print(plot_metric)
	
	plt.show()

dataset='cv'
load_metrics=True
#mode='global'
mode='each_date'
if dataset=='cv':
	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]

##		'prediction_DenseNetTimeDistributed_128x2_redoing.npy']
		##'prediction_ConvLSTM_seq2seq_loneish.npy',
		##'prediction_ConvLSTM_seq2seq_bi_loneish.npy',
		#'prediction_ConvLSTM_seq2seq_bi_60x2_loneish.npy',
		#'prediction_FCN_ConvLSTM_seq2seq_bi_skip_loneish.npy',
		#'prediction_DenseNetTimeDistributed_blockgoer.npy',
		#'prediction_DenseNetTimeDistributed_128x2_filtersizefix2.npy']
elif dataset=='lm':

	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],
		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy'],]

if load_metrics==False:
	experiment_metrics=experiment_groups_analyze(dataset,experiment_groups,
		mode=mode)
	np.save("experiment_metrics_"+dataset+".npy",experiment_metrics)

else:
	experiment_metrics=np.load("experiment_metrics_"+dataset+".npy")
	print("Difference F1 in percentage",np.average(experiment_metrics[2]['f1_score']-experiment_metrics[1]['f1_score']))
	print("Difference OA in percentage",np.average(experiment_metrics[2]['overall_acc']-experiment_metrics[1]['overall_acc']))


if mode=='each_date':
	experiments_plot(experiment_metrics,experiment_groups[0],dataset)

#metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])]


