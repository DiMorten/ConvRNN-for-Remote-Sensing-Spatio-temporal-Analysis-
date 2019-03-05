
id=$1
model=$2

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
cd ..
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
results_path='../results/seq2seq_ignorelabel/cv/'
mv prediction.npy "${results_path}prediction_${model}_${id}.npy"
cd scripts

