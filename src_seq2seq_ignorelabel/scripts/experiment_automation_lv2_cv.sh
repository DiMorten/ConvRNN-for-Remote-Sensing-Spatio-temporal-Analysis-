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


##id='v3plus3'

#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'deeplabv3plus' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication

##dataset='cv'
##. patches_extract.sh $dataset
##. experiment_automation.sh $id 'deeplabv3plus' $dataset  # gonna test balancing after replication
##. experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication

# id='times'


# dataset='cv'
# #. patches_extract.sh $dataset
# #. experiment_automation.sh $id 'deeplabv3plus' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'deeplab_rs_multiscale' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'deeplab_rs' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'deeplabv3' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'pyramid_dilated_bconvlstm' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication


# id='2convins2'
# dataset='cv'
# . patches_extract.sh $dataset
# . experiment_automation.sh $id 'BUnetConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BAtrousConvLSTM' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'BAtrousASPPConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnetAtrousConvLSTM' $dataset  # gonna test balancing after replication
# dataset='lm'
# . patches_extract.sh $dataset
# . experiment_automation.sh $id 'BUnetConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BAtrousConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnetAtrousConvLSTM' $dataset  # gonna test balancing after replication



# id='2convins2'
# dataset='cv'
# #. patches_extract.sh $dataset
# #. experiment_automation.sh $id 'BUnetAtrousConvLSTM_v3p' $dataset
# . experiment_automation.sh $id 'BUnetAtrousConvLSTM' $dataset  # gonna test balancing after replication

# id='raulapproved'
# #dataset='cv'
# #. patches_extract.sh $dataset
# #. experiment_automation.sh $id 'BUnetConvLSTM' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'BUnetAtrousConvLSTM' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'BUnetAtrousConvLSTM_v3p' $dataset  # gonna test balancing after replication
# #. experiment_automation.sh $id 'BAtrousConvLSTM' $dataset  # gonna test balancing after replication
# dataset='lm'
# #. patches_extract.sh $dataset

# . experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnetConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BAtrousConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnetAtrousConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnetAtrousConvLSTM_v3p' $dataset  # gonna test balancing after replication


#. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'BUnet2ConvLSTM' $dataset  # gonna test balancing after replication
#id='repeating1'
#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # gonna test balancing after replication


id='repeating_timey'

#dataset='cv'
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'BAtrousConvLSTM' $dataset  # gonna test balancing after replication

#dataset='lm'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # Unet5 uses 1 conv. in
#. experiment_automation.sh $id 'BUnet5ConvLSTM' $dataset  # Unet5 uses 1 conv. in
#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'BUnet5ConvLSTM' $dataset  # gonna test balancing after replication
dataset='cv'
. patches_extract.sh $dataset
#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2_inconv' $dataset  # Unet5 uses 1 conv. in
. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset  # Unet5 uses 1 conv. in
. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # Unet5 uses 1 conv. in
. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  # Unet5 uses 1 conv. in
. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication
. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # Unet5 uses 1 conv. in

#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2_inconv' $dataset  # Unet5 uses 1 conv. in

#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset  # gonna test balancing after replication

#dataset='cv'
#. patches_extract.sh $dataset
#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2_3blocks' $dataset  # gonna test balancing after replication


