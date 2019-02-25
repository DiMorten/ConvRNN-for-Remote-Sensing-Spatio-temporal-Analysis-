 
#!/bin/bash


FILE1=$1 # test.png
imgur-uploader $FILE1 > log1.txt

tr '\n' ' ' < log1.txt > log2.txt
cmd="s,Uploading file ${FILE1} File uploaded - see your image at ,,g"
echo 'cmd: '${cmd} 
sed -e "s,Uploading file ${FILE1} File uploaded - see your image at ,,g" log2.txt > log3.txt
#sed -e 's,Uploading file ../campo_verde_ConvLSTM_eyesight.png File uploaded - see your image at ,,g' log2.txt > log3.txt

url=`cat log3.txt`
echo "url: ${url}"
python tl_ifttt_request.py --url=$url
cd -
