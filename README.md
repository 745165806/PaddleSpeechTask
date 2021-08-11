# PaddleSpeechTask
A speech library to deal with a series of related front-end and back-end tasks  


## 中/英文文本加标点任务 punctuation restoration：
##### 数据集
中文数据来源：
- 1.iwlst
- 2.平凡的世界

英文数据来源
- 1.iwlst2012


##### 模型：
1.BLSTM模型

2.BertLinear模型

3.BertBLSTM模型

##### 实验结果
* 基线：
- 中文 BertLinear Model
  
进入./example/chinese目录 

运行 run.sh 1 5 conf/bertLinear.yaml 1


基线结果：
    |            |comma 。 |period ， |question ？ |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |0.576450 |0.808449  |0.696598    |0.693799 |
    |recall      |0.391214 |0.587661  |0.736626    |0.571833 |
    |F1          |0.466102 |0.690597  |0.716000    |0.620900 |