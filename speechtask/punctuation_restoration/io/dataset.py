# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
import ujson
import codecs
import random


from paddle.io import Dataset
import paddle
import os
import numpy as np

from speechtask.punctuation_restoration.utils.log import Log
# from speechtask.punctuation_restoration.utils.punct_prepro import load_dataset

__all__ = [
    "PuncDataset",
]

logger = Log(__name__).getlog()

class PuncDataset(Dataset):
    """Representing a Dataset
    superclass
    ----------
    data.Dataset :
        Dataset is a abstract class, representing the real data.
    """
    def __init__(self, train_path, vocab_path, punc_path, seq_len=100):
        # 检查文件是否存在
        print(train_path)
        print(vocab_path)
        assert os.path.exists(train_path), "train文件不存在"
        assert os.path.exists(vocab_path), "词典文件不存在"
        assert os.path.exists(punc_path), "标点文件不存在"
        self.seq_len = seq_len

        self.word2id = self.load_vocab(
            vocab_path,
            extra_word_list=['<UNK>', '<END>']
        )
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.punc2id = self.load_vocab(
            punc_path,
            extra_word_list=[" "]
        )
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        # print(self.txt_seqs[:10])
        # with open('./txt_seq', 'w', encoding='utf-8') as w:
        #     print(self.txt_seqs, file=w)
        self.preprocess(self.txt_seqs)
        print('---punc-')
        print(self.punc2id)

        
    def __len__(self):
        """return the sentence nums in .txt
        """
        return self.in_len

    def __getitem__(self, index):
        """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)
        Parameters
        ----------
        index : int 索引
        """
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[], encoding='utf-8'):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i+n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_data = []
        label = []
        input_r = []
        label_r = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
        count = 0
        length = len(txt_seqs)
        for token in txt_seqs:
            count += 1
            if count == length:
                break
            if token in self.punc2id:
                continue
            punc = txt_seqs[count]
            if punc not in self.punc2id:
                # print('标点{}：'.format(count), self.punc2id[" "])
                label.append(self.punc2id[" "])
                input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(' ')
            else:
                # print('标点{}：'.format(count), self.punc2id[punc])
                label.append(self.punc2id[punc])
                input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(punc)
        # with open(os.path.join(self.save_dir, 'input_lbl'), 'w', encoding='utf-8') as w:
        #     print('输入数据是：', input_r, file=w)
        #     print('输出标签是：', label_r, file=w)
        # with open(os.path.join(self.save_dir, './input_lbl_id'), 'w', encoding='utf-8') as w:
        #     print('输入数据是：', input_data, file=w)
        #     print('输出标签是：', label, file=w)
        if len(input_data) != len(label):
            assert 'error: length input_data != label'
        # code below is for using 100 as a hidden size
        print(len(input_data))
        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]

        self.input_data = paddle.to_tensor(np.array(input_data, dtype='int64').reshape(-1, self.seq_len))
        self.label = paddle.to_tensor(np.array(label, dtype='int64').reshape(-1, self.seq_len))




# class PRDataset(Dataset):
#     def __init__(self, data_path):
#         """Manifest Dataset

#         Args:
#             manifest_path (str): manifest josn file path
#             max_input_len ([type], optional): maximum output seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to float('inf').
#             min_input_len (float, optional): minimum input seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to 0.0.
#             max_output_len (float, optional): maximum input seq length, in modeling units. Defaults to 500.0.
#             min_output_len (float, optional): minimum input seq length, in modeling units. Defaults to 0.0.
#             max_output_input_ratio (float, optional): maximum output seq length/output seq length ratio. Defaults to 10.0.
#             min_output_input_ratio (float, optional): minimum output seq length/output seq length ratio. Defaults to 0.05.
        
#         """
#         super().__init__()

#         # read manifest
#         self.load_dataset(data_path)

#     def load_dataset(self, data_path):
#         with codecs.open(data_path, mode='r', encoding='utf-8') as f:
#             self.dataset = ujson.load(f)

#     def __len__(self):
#         return len(self._manifest)

#     def __getitem__(self, idx):
#         instance = self._manifest[idx]
#         return instance["utt"], instance["feat"], instance["text"]

if __name__ == '__main__':
    dataset=PuncDataset()