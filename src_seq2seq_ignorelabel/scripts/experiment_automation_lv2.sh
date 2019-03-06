KERAS_BACKEND=tensorflow
id='lmish'

#dataset='campo_verde'
dataset='lm'


# ==== EXTRACT PATCHES
#. patches_extract.sh $dataset
# ===== USE MODEL

. experiment_automation.sh $id 'DenseNetTimeDistributed' $dataset 
. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset
. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi_60x2' $dataset
. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset




