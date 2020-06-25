KERAS_BACKEND=tensorflow



# id='repeating_timey4'

# dataset='lm'
# . patches_extract.sh $dataset
# . experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # Unet5 uses 1 conv. in
# . experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  # Unet5 uses 1 conv. in
# . experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # Unet5 uses 1 conv. in

# dataset='cv'
# . patches_extract.sh $dataset

# . experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # gonna test balancing after replication
# . experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # Unet5 uses 1 conv. in
# . experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication



#id='first7images'


dataset='cv'
#. patches_extract.sh $dataset
#======================== REGULAR BUNETCONVLSTM ======================================= #
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset
#id='first7images4'
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset
id='first5images3'
t_step=5
#. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $t_step
. experiment_automation.sh $id 'Unet4ConvLSTM' $dataset $t_step

#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset  # gonna test balancing after replication

# count=0
# id_iteration="${id}${count}"
# echo "Starting id: ${id_iteration}"
# cd ..
# #python hyperparameter_handler.py --model_id="dense" --model_params 128 2 80 2
# python hyperparameter_handler.py --model_id="convlstm" --model_params 128 2 80 2

# cd scripts/
# . experiment_automation.sh $id_iteration 'DenseNetTimeDistributed_128x2' $dataset  

# count=1
# id_iteration="${id}${count}"
# echo "Starting id: ${id_iteration}"
# cd ..
# python hyperparameter_handler.py --model_id="dense" --model_params 128 2 64 3
# cd scripts/

# . experiment_automation.sh $id_iteration 'DenseNetTimeDistributed_128x2' $dataset  

# count=2
# id_iteration="${id}${count}"
# echo "Starting id: ${id_iteration}"
# cd ..
# python hyperparameter_handler.py --model_id="dense" --model_params 128 2 64 1
# cd scripts/
# . experiment_automation.sh $id_iteration 'DenseNetTimeDistributed_128x2' $dataset  
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset  