KERAS_BACKEND=tensorflow

#dataset='campo_verde'

#dataset='lm'

#. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset 



#dataset='cv_seq1'

# ==== EXTRACT PATCHES
#. patches_extract.sh $dataset
# ===== USE MODEL

#. experiment_automation.sh $id 'DenseNetTimeDistributed' $dataset 
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi_60x2' $dataset
#. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset

##. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset
##. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
##. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset




# id='redoing3'

# dataset='cv'
# . patches_extract.sh $dataset

# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
# . experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset


# dataset='lm'
# . patches_extract.sh $dataset

# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
# . experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset

id='dilated'
dataset='cv'
#. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication
. experiment_automation.sh $id 'unet_bconvlstm' $dataset  # gonna test balancing after replication

#. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication