
id=$1
model=$2
dataset=$3 # could be cv or lem
#fileid=$4 # like prediction_densenet_.....npy
if [ "$dataset" == "cv_seq1" ]
then
	filename="campo_verde"
	results_path='../results/seq2seq_ignorelabel/cv/'
	summary_save_path='../results/seq2seq_ignorelabel/summary/cv/'
	dataset_path="../../../deep_learning/LSTM-Final-Project/cv_data/"
	sequence_len=7
	class_n=12
elif [ "$dataset" == "cv" ]
then
	filename="campo_verde"
	results_path='../results/seq2seq_ignorelabel/cv/'
	summary_save_path='../results/seq2seq_ignorelabel/summary/cv/'
	dataset_path="../../../deep_learning/LSTM-Final-Project/cv_data/"
	sequence_len=14
	class_n=12

else
	filename="lm"
	results_path='../results/seq2seq_ignorelabel/lm/'
	summary_save_path='../results/seq2seq_ignorelabel/summary/lm/'
	dataset_path="../../../deep_learning/LSTM-Final-Project/lm_data/"
	sequence_len=13
	class_n=15 # 14+bcknd
fi


#id="blockgoer"
rm -f log1.txt
rm -f log2.txt
rm -f log3.txt
#model='FCN_ConvLSTM'
##model='ConvLSTM_DenseNet'
#model='FCN_ConvLSTM2'
#model='BiConvLSTM_DenseNet'
##model='ConvLSTM_seq2seq'
##model='FCN_ConvLSTM_seq2seq_bi'
##model='FCN_ConvLSTM_seq2seq_bi_skip'

##model='DenseNetTimeDistributed'
#model='ConvLSTM_seq2seq_bi' # russworm bi .
# ============== EXECUTE EXPERIMENT ===============
#python main_hdd.py -pl=32 -pstr=32 -psts=32 -path=$dataset_path -tl=$sequence_len -cn=$class_n -chn=2 -mdl=$model
echo "${filename}_${model}_${id}"

# ========= TAKE SCREENSHOT ===============
im_name="../results/seq2seq_ignorelabel/${dataset}/prediction_${model}_${id}.npy"
echo "Doing inference for file: ${im_name}. Dataset: ${dataset}"
cd ..
python inference_time_measure.py -pl=32 -pstr=32 -psts=32 -bstr=16 -bsts=16 -path=$dataset_path -tl=$sequence_len -cn=$class_n -chn=2 -mdl=$model -wfl=$im_name


cd scripts/