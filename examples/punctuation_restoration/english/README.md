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
  - 实验配置：conf/train_conf/bertBLSTM_base_en.yaml
  - 测试结果：exp/bertBLSTM_base_en/checkpoints/4 

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  |0.617347   |0.700883   |0.641509   |0.653246  |
    |Recall     |0.678822   |0.804816   |0.739130   |0.740923  |
    |F1         |0.646627   |0.749263   |0.686869   |0.694253  |  
