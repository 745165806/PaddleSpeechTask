from functools import partial
import numpy as np

import paddle
import paddle.nn as nn
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification


class BertLinearPunc(nn.Layer):
    def __init__(self, pretrained_token="bert-base-uncased", output_size=4, dropout=0.2, bert_size=768, hiddensize=1568):
        super(BertLinearPunc, self).__init__()
        self.output_size = output_size
        self.bert = BertForTokenClassification.from_pretrained(pretrained_token, num_classes=bert_size)
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bert_size, hiddensize)
        self.dropout2 = nn.Dropout(dropout)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(hiddensize, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print('input :', x.shape)
        x = self.bert(x) #[0]
        # print('after bert :', x.shape)
    

        x = self.fc1(self.dropout1(x))
        x = self.fc2(self.relu(self.dropout2(x)))
        x = paddle.reshape(x, shape=[-1,self.output_size])
        # print('after fc :', x.shape)

        logit = self.softmax(x)
        # print('after softmax :', logit.shape)
        
        return x, logit

if __name__ == '__main__':
    print('start model')
    model = BertLinearPunc()
    x=paddle.randint(low=0, high=40,shape=[2, 5])
    print(x)
    y, logit = model(x)