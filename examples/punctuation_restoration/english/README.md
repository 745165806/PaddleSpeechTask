# 英文实验例程
## 测试数据：
- IWLST2012英文：test2011

## 运行代码
- 运行 `run.sh 0 0 conf/train_conf/bertBLSTM_base_en.yaml 1 conf/data_conf/english.yaml `

## 实验结果：
- BertBLSTM
  - 实验配置：conf/train_conf/bertBLSTM_base_en.yaml
  - 测试结果： epoch99

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  | 0.680224  | 0.751584  | 0.652174  | 0.694661 |
    |Recall     | 0.598765  | 0.692757  | 0.625000  | 0.638841 |
    |F1         | 0.636901  | 0.720973  | 0.638298  | 0.665390 |  