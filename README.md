# PaddleSpeechTask
A speech library to deal with a series of related front-end and back-end tasks  

## 环境
- python==3.6.13
- paddle==2.1.1

## 中/英文文本加标点任务 punctuation restoration：

### 数据集: data
- 中文数据来源：data/chinese  
1.iwlst2012zh  
2.平凡的世界

-  英文数据来源: data/english  
1.iwlst2012en

- iwlst2012数据获取过程见data/README.md

### 模型：speechtask/punctuation_restoration/model
1.BLSTM模型

2.BertLinear模型

3.BertBLSTM模型

### 实验结果
- 基线：
 
- 英文 BertBLSTM Model  pretrain_model:bert-base-uncased
  
进入./example/english目录 模型.exp/bertBLSTM_base_en/checkpoints/avg_1  (4)

基线结果：

    |            |comma ,  |period .  |question ?  |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |0.672705 |0.844527  |0.717391    |0.744875 |
    |recall      |0.679268 |0.781358  |0.647059    |0.702562 |
    |F1          |0.675971 |0.811715  |0.680412    |0.722700 |

