
id="eyesight"

model='ConvLSTM'
cd ..
python main.py -pl=32 -pstr=32 -psts=32 -path="../../../deep_learning/LSTM-Final-Project/cv_data/" -tl=7 -cn=12 -chn=2 -mdl=$model
echo "campo_verde_${model}_${id}"
rm log1.txt
rm log2.txt
rm log3.txt
im_name="campo_verde_${model}_${id}.png"
shutter -f -o $im_name -e

cd scripts
path="../${im_name}"
echo "${path}"
. ifttt_send.sh $path
cd scripts