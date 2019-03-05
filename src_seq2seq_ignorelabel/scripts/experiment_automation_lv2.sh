KERAS_BACKEND=tensorflow
id='loneish'

#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' 
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' 
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi_60x2' 
. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' 




