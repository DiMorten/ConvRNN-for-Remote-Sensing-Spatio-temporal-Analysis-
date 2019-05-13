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

# id='v3plus'

# dataset='cv'

# #. patches_extract.sh $dataset
# #. experiment_automation.sh $id 'deeplab_rs' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'deeplabv3' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset
# . experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication

# #. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication

# dataset='lm'
# . patches_extract.sh $dataset
# #. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset
# #. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'deeplab_rs' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'deeplabv3' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication


id='v3plus2'

#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'deeplabv3plus' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication

dataset='lm'
. patches_extract.sh $dataset
#. experiment_automation.sh $id 'deeplabv3plus' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication
. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication
. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication
