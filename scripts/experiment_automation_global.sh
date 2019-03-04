
id="blockgoer"
rm -f log1.txt
rm -f log2.txt
rm -f log3.txt
#model='FCN_ConvLSTM'
#model='ConvLSTM_DenseNet'
#model='FCN_ConvLSTM2'

model='ConvLSTM_seq2seq' # russworm bi .
#cd ~/Jorg/deep_learning/LSTM-Final-Project/src_seq2seq
#python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

#cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq

# ============== EXECUTE EXPERIMENT ===============
cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq
#cd ..
python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2 -mdl=$model

echo "campo_verde_${model}_${id}"


# ========= TAKE SCREENSHOT ===============
im_name="campo_verde_${model}_${id}.png"
wmctrl -a konsole
shutter -f -o $im_name -e

# ============== SEND IMAGE TO FACEBOOK MESSENGER =========
cd scripts
path="../${im_name}"
echo "${path}"
. ifttt_send.sh $path

# =============== MOVE PREDICTIONS TO RESULT FOLDER ======
results_path='../results/seq2seq/cv/'
cp prediction.npy "${results_path}prediction_${model}_${id}.npy"






model='DenseNetTimeDistributed' # russworm bi .
#cd ~/Jorg/deep_learning/LSTM-Final-Project/src_seq2seq
#python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

#cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq

# ============== EXECUTE EXPERIMENT ===============
cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq
#cd ..
python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2 -mdl=$model

echo "campo_verde_${model}_${id}"


# ========= TAKE SCREENSHOT ===============
im_name="campo_verde_${model}_${id}.png"
wmctrl -a konsole
shutter -f -o $im_name -e

# ============== SEND IMAGE TO FACEBOOK MESSENGER =========
cd scripts
path="../${im_name}"
echo "${path}"
. ifttt_send.sh $path

# =============== MOVE PREDICTIONS TO RESULT FOLDER ======
results_path='../results/seq2seq/cv/'
cp prediction.npy "${results_path}prediction_${model}_${id}.npy"







model='FCN_ConvLSTM_seq2seq_bi_skip' # russworm bi .
#cd ~/Jorg/deep_learning/LSTM-Final-Project/src_seq2seq
#python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

#cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq

# ============== EXECUTE EXPERIMENT ===============
cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq
#cd ..
python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2 -mdl=$model

echo "campo_verde_${model}_${id}"


# ========= TAKE SCREENSHOT ===============
im_name="campo_verde_${model}_${id}.png"
wmctrl -a konsole
shutter -f -o $im_name -e

# ============== SEND IMAGE TO FACEBOOK MESSENGER =========
cd scripts
path="../${im_name}"
echo "${path}"
. ifttt_send.sh $path

# =============== MOVE PREDICTIONS TO RESULT FOLDER ======
results_path='../results/seq2seq/cv/'
cp prediction.npy "${results_path}prediction_${model}_${id}.npy"





model='ConvLSTM_seq2seq_bi' # russworm bi .
#cd ~/Jorg/deep_learning/LSTM-Final-Project/src_seq2seq
#python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

#cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq

# ============== EXECUTE EXPERIMENT ===============
cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq
#cd ..
python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2 -mdl=$model

echo "campo_verde_${model}_${id}"


# ========= TAKE SCREENSHOT ===============
im_name="campo_verde_${model}_${id}.png"
wmctrl -a konsole
shutter -f -o $im_name -e

# ============== SEND IMAGE TO FACEBOOK MESSENGER =========
cd scripts
path="../${im_name}"
echo "${path}"
. ifttt_send.sh $path

# =============== MOVE PREDICTIONS TO RESULT FOLDER ======
results_path='../results/seq2seq/cv/'
cp prediction.npy "${results_path}prediction_${model}_${id}.npy"

