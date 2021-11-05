import os
import re
file_name='/data4/mahaoxin/PaddleSpeechTask/examples/punctuation_restoration/chinese/data/iwslt2012_zh/dev'
output=open('/data4/mahaoxin/PaddleSpeechTask/examples/punctuation_restoration/chinese/data/iwslt2012_zh_revise/dev_revise','w', encoding='utf-8')
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
    line = re.sub('[a-zA-Z]','',line)
    line = line.replace('（ ','').replace('） ','')
    line = line.replace('“ ','').replace('” ','').replace('‘ ','').replace(') ','').replace('（ ','')
    line = line.replace('- - ','，').replace('： ','，').replace('% ','')
    output.write(line)
    # output.write("\n")
        # print(word, punc)
    # else:
    #     print('err')
    #     print(i)
    #     print(line)
output.close()