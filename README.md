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

### iwlst2012数据获取过程：
- 1 官网IWSLT2012:  
  train：https://wit3.fbk.eu/2012-03  
  test: https://wit3.fbk.eu/2012-03-b  
  找到train数据下载链接：https://drive.google.com/file/d/1aTW5gG2xCZbfNy5rOzG7iSJ7TVGywxcx/view?usp=sharing；  
  test数据下载链接：https://drive.google.com/file/d/1974h-vndIdVvJZEz4S3t4gkmaGFARuok/view?usp=sharing

- 2 .tgz文件解压  
  linux系统下，命令为：`tar zxvf 2012-03.tgz`;  `tar zxvf 2012-03-test.tgz`

- 3 cd 2012-03 进入训练集目录，生成训练和验证文本    
  `cd texts`   
  `cd en` 进入英文IWSLT训练集  
  `cd fr`  
  `tar zxvf en-fr.tgz`  
  `cd en-fr`
  
- 3.1 生成IWSLT2012英文训练集train：  
  在en-fr文件夹下：  
  `cp train.tags.en-fr.en iwslt2012_train_en`  
  `vim iwslt2012_train_en` 进入iwslt2012_train_en文件  
  删除1-6行以及最后一行，即只保留<transcript>内容，得到IWSLT2012英文训练文本iwslt2010_train_en  

- 3.2 生成IWSLT2012英文验证集dev： 
  在en-fr文件夹下：   
  `cat IWSLT12.TALK.dev2010.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2010_dev_en`   
  得到IWSLT2012英文验证文本iwslt2010_dev_en  

- 3.3 同样方法生成IWSLT2012中文训练集和验证集：  
  返回2012-03/texts目录; cd zh; cd en; tar zxvf zh-en.tgz; cd zh-en; 进入zh-en文件夹  
  `cp train.tags.zh-en.zh iwslt2012_train_zh`  
  `vim iwslt2012_train_zh` 进入iwslt2012_train_zh文件  
  删除1-6行以及最后一行，即只保留<transcript>内容，得到IWSLT2012中文训练文本iwslt2010_train_zh  
  `cat IWSLT12.TALK.dev2010.zh-en.zh.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2010_dev_zh` ; 得到IWSLT2012中文验证文本iwslt2010_dev_zh  

- 4 退出2012-03训练集目录，进入2012-03-test目录，生成测试文本  
  `cd 2012-03-test`;  
  `cd texts`;  
  `cd en/fr`;   
  `tar zxvf en-fr.tgz`;   
  `cd en-fr`;  
  `cat IWSLT12.TED.MT.tst2011.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2011_test_en` ; 得到IWSLT2012英文测试文本iwslt2011_test_en; 也是IWSLT2012比赛 MT任务的官方评测文本   
  `cat IWSLT12.TED.MT.tst2012.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2012_test_en` ; 得到IWSLT2012英文测试文本iwslt2012_test_en  

  返回2012-03-test目录，同样方法处理得到中文的相应2011和2012年的测试文本iwslt2011_test_zh, iwslt2011_test_zh  

- 5 将处理后的iwslt201*_*_*文本移动至相应的PaddleSpeechTask/data/english/iwslt2012_en和PaddleSpeechTask/data/chinese/iwslt2012_zh
  

### 模型：speechtask/punctuation_restoration/model
1.BLSTM模型

2.BertLinear模型

3.BertBLSTM模型

### 实验结果
* 基线：
- 中文 BertLinear Model
  
进入./example/chinese目录 

运行 run.sh 1 0 conf/bertLinear.yaml 1

基线结果：

    |            |comma ， |period 。 |question ？ |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |0.576450 |0.808449  |0.696598    |0.693799 |
    |recall      |0.391214 |0.587661  |0.736626    |0.571833 |
    |F1          |0.466102 |0.690597  |0.716000    |0.620900 |

- 中文 BertBLSTM Model
  
进入./example/chinese目录 模型./exp/bertBLSTM_ch/checkpoints/49

运行 run.sh 1 0 conf/train_conf/bertBLSTM_ch.yaml 1

基线结果：

    |            |comma ， |period 。 |question ？ |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |0.532847 |0.777691  |0.669261    |0.659933 |
    |recall      |0.402729 |0.563231  |0.674510    |0.546824 |
    |F1          |0.458740 |0.653312  |0.671875    |0.594642 |


- 英文 BertBLSTM Model  pretrain_model:bert-base-uncased
  
进入./example/english目录 模型.exp/bertBLSTM_base_en/checkpoints/avg_1  (4)

基线结果：

    |            |comma ,  |period .  |question ?  |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |0.672705 |0.844527  |0.717391    |0.744875 |
    |recall      |0.679268 |0.781358  |0.647059    |0.702562 |
    |F1          |0.675971 |0.811715  |0.680412    |0.722700 |


？？？？？？？还没有实验
- 英文 BertBLSTM Model  pretrain_model:bert-large-uncased
  
进入./example/english目录 模型./exp/bertBLSTM_ch/checkpoints/49

基线结果：

    |            |comma ,  |period .  |question ?  |overall  |
    |------------|---------|----------|------------|---------|
    |precision   |
    |recall      |
    |F1          |

