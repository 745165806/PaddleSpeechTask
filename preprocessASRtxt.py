import os
file_name='/data4/mahaoxin/PaddleSpeechTask/biaobei.txt'
output=open('/data4/mahaoxin/PaddleSpeechTask/biaobei_asr','w', encoding='utf-8')
f=open(file_name,'r',encoding='utf-8')
pun_dic={"COMMA": ",", "PERIOD": ".", "QUESTION": "?", "O": ""}
# pun_dic= {
#         'O': '',
#         '，': "，",
#         '。': '。',
#         '？': '？',
#     }
voc={}
i=0
for line in f.readlines():
    if(line[0]!='0'):continue
    line=line.strip()[7:]
    line=line.replace('#','')
    line=line.replace('1','')
    line=line.replace('2','')
    line=line.replace('3','')
    line=line.replace('4','')
    line=line.replace('5','')
    line=line.replace('6','')
    output.write(line)
    output.write('\n')
output.close()