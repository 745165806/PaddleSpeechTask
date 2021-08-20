# 英文实验例程
## 测试数据：
- IWLST2012英文：test2011

## 运行代码
- 运行 `run.sh 0 0 conf/train_conf/bertBLSTM_base_en.yaml 1 conf/data_conf/english.yaml `


## 相关论文实验结果：
> * Nagy, Attila, Bence Bial, and Judit Ács. "Automatic punctuation restoration with BERT models." arXiv preprint arXiv:2101.07343 (2021)*  
> 


## 实验结果：
- BertBLSTM
  - 实验配置：conf/train_conf/bertLinear_en.yaml
  - 测试结果：exp/bertLinear_en/checkpoints/3.pdparams

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  |0.673421   |0.647972   |0.782609   |0.701334  |
    |Recall     |0.721584   |0.856328   |0.720000   |0.765971  |
    |F1         |0.696671   |0.737721   |0.750000   |0.728131  |  
